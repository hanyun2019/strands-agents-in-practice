import subprocess
import os
import shutil
import platform
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP

# Data classes for dependency checking
@dataclass
class LaTeXStatus:
    latex_available: bool
    pdflatex_path: Optional[str] = None
    xelatex_path: Optional[str] = None
    lualatex_path: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class SystemDependencies:
    latex: bool
    cairo: bool
    pkg_config: bool
    ffmpeg: bool
    manim: bool
    missing_packages: List[str] = None
    
    def __post_init__(self):
        if self.missing_packages is None:
            self.missing_packages = []

def _handle_manim_error(stderr: str, stdout: str, manim_code: str) -> str:
    """Enhanced error handling for Manim execution failures with LaTeX-specific detection"""
    error_output = stderr + "\n" + stdout
    
    # Check for LaTeX-specific errors
    latex_error_indicators = [
        "LaTeX Error:",
        "! LaTeX Error:",
        "pdflatex",
        "xelatex", 
        "lualatex",
        "tex error",
        "Missing $ inserted",
        "Undefined control sequence",
        "Package amsmath Error:",
        "! Missing $ inserted",
        "! Undefined control sequence",
        "! Package amsmath Error:",
        "LaTeX compilation failed"
    ]
    
    # Check if this is a LaTeX-related error
    is_latex_error = any(indicator.lower() in error_output.lower() for indicator in latex_error_indicators)
    
    if is_latex_error:
        return _handle_latex_error(error_output, manim_code)
    
    # Check for other common Manim errors
    if "ModuleNotFoundError" in error_output:
        return _handle_module_error(error_output)
    elif "ImportError" in error_output:
        return _handle_import_error(error_output)
    elif "SyntaxError" in error_output:
        return _handle_syntax_error(error_output)
    elif "AttributeError" in error_output:
        return _handle_attribute_error(error_output)
    else:
        return f"Execution failed with error:\n{error_output}\n\nFor troubleshooting help, run check_rendering_capabilities() to verify your system setup."


def _handle_latex_error(error_output: str, manim_code: str) -> str:
    """Handle LaTeX-specific errors with detailed diagnostics and fallback suggestions"""
    # Check if LaTeX is available
    latex_status = DependencyChecker.check_latex_installation()
    
    error_message = ["LaTeX Rendering Error Detected"]
    error_message.append("=" * 40)
    
    # Provide specific LaTeX error analysis
    if "! LaTeX Error:" in error_output:
        error_message.append("LaTeX compilation failed with the following error:")
        # Extract the specific LaTeX error
        lines = error_output.split('\n')
        for i, line in enumerate(lines):
            if "! LaTeX Error:" in line:
                error_message.append(f"  {line}")
                # Include next few lines for context
                for j in range(1, min(4, len(lines) - i)):
                    if lines[i + j].strip():
                        error_message.append(f"  {lines[i + j]}")
                break
    elif "Undefined control sequence" in error_output:
        error_message.append("LaTeX Error: Undefined control sequence detected.")
        error_message.append("This usually means you're using a LaTeX command that's not recognized.")
    elif "Missing $ inserted" in error_output:
        error_message.append("LaTeX Error: Missing $ inserted.")
        error_message.append("This usually means mathematical expressions need to be wrapped in $ symbols.")
    else:
        error_message.append("LaTeX compilation failed. Full error output:")
        error_message.append(error_output)
    
    error_message.append("")
    
    # Check LaTeX availability and provide appropriate guidance
    if not latex_status.latex_available:
        error_message.append("DIAGNOSIS: LaTeX is not installed on your system.")
        error_message.append("")
        error_message.append("SOLUTION OPTIONS:")
        error_message.append("1. Install LaTeX to enable full mathematical rendering:")
        missing_deps = ["LaTeX"]
        instructions = DependencyChecker.get_installation_instructions(missing_deps)
        error_message.append(f"   {instructions}")
        error_message.append("")
        error_message.append("2. Use fallback rendering by modifying your code:")
        error_message.append("   - Replace Tex() with Text() for simple text")
        error_message.append("   - Replace MathTex() with Text() for basic mathematical expressions")
        error_message.append("   - Use Unicode symbols instead of LaTeX commands (e.g., 'α' instead of r'\\alpha')")
    else:
        error_message.append("DIAGNOSIS: LaTeX is installed but compilation failed.")
        error_message.append("")
        error_message.append("SOLUTION OPTIONS:")
        error_message.append("1. Check your LaTeX syntax - common issues:")
        error_message.append("   - Ensure mathematical expressions are properly escaped")
        error_message.append("   - Use raw strings (r'...') for LaTeX commands")
        error_message.append("   - Verify all LaTeX packages are available")
        error_message.append("")
        error_message.append("2. Try fallback rendering:")
        error_message.append("   - Replace complex LaTeX with simpler Text() objects")
        error_message.append("   - Use MathTex() instead of Tex() for mathematical expressions")
        error_message.append("   - Break complex expressions into simpler parts")
    
    # Analyze the code for common LaTeX issues
    latex_suggestions = _analyze_latex_code_issues(manim_code)
    if latex_suggestions:
        error_message.append("")
        error_message.append("CODE ANALYSIS SUGGESTIONS:")
        error_message.extend(latex_suggestions)
    
    error_message.append("")
    error_message.append("For more detailed system information, run: check_rendering_capabilities()")
    
    return "\n".join(error_message)


def _analyze_latex_code_issues(manim_code: str) -> List[str]:
    """Analyze Manim code for common LaTeX issues and provide specific suggestions"""
    suggestions = []
    
    # Check for common LaTeX issues
    if "Tex(" in manim_code and not any(line.strip().startswith("r'") or line.strip().startswith('r"') for line in manim_code.split('\n') if "Tex(" in line):
        suggestions.append("- Consider using raw strings (r'...') for LaTeX expressions to avoid escape sequence issues")
    
    if "\\alpha" in manim_code or "\\beta" in manim_code or "\\gamma" in manim_code:
        suggestions.append("- Greek letters detected: ensure they're in raw strings or consider using Unicode: α, β, γ")
    
    if "\\frac" in manim_code:
        suggestions.append("- Fractions detected: ensure proper LaTeX syntax like r'\\frac{numerator}{denominator}'")
    
    if "\\sum" in manim_code or "\\int" in manim_code:
        suggestions.append("- Mathematical operators detected: ensure they're properly formatted in LaTeX")
    
    if "MathTex(" in manim_code:
        suggestions.append("- Using MathTex: this requires LaTeX. Consider Text() for simple expressions if LaTeX fails")
    
    if "$" in manim_code and "Tex(" in manim_code:
        suggestions.append("- Avoid mixing $ symbols with Tex() - Tex() handles math mode automatically")
    
    return suggestions


def _handle_module_error(error_output: str) -> str:
    """Handle Python module import errors"""
    return f"Module Import Error:\n{error_output}\n\nSolution: Install the missing Python package using pip install <package_name>"


def _handle_import_error(error_output: str) -> str:
    """Handle Python import errors"""
    return f"Import Error:\n{error_output}\n\nSolution: Check that all required modules are installed and accessible"


def _handle_syntax_error(error_output: str) -> str:
    """Handle Python syntax errors"""
    return f"Python Syntax Error:\n{error_output}\n\nSolution: Check your Python code syntax, particularly indentation and brackets"


def _handle_attribute_error(error_output: str) -> str:
    """Handle Python attribute errors"""
    return f"Attribute Error:\n{error_output}\n\nSolution: Check that you're using the correct Manim object methods and attributes"


class DependencyChecker:
    """Utility class for checking system dependencies required for LaTeX rendering"""
    
    @staticmethod
    def check_command_available(command: str) -> Tuple[bool, Optional[str]]:
        """Check if a command is available in the system PATH"""
        try:
            result = subprocess.run(
                ["which", command] if platform.system() != "Windows" else ["where", command],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return True, result.stdout.strip()
            return False, None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, None
    
    @staticmethod
    def check_latex_installation() -> LaTeXStatus:
        """Check for LaTeX installation and return detailed status"""
        pdflatex_available, pdflatex_path = DependencyChecker.check_command_available("pdflatex")
        xelatex_available, xelatex_path = DependencyChecker.check_command_available("xelatex")
        lualatex_available, lualatex_path = DependencyChecker.check_command_available("lualatex")
        
        latex_available = pdflatex_available or xelatex_available or lualatex_available
        
        error_message = None
        if not latex_available:
            error_message = "No LaTeX distribution found. Please install LaTeX to enable mathematical rendering."
        
        return LaTeXStatus(
            latex_available=latex_available,
            pdflatex_path=pdflatex_path,
            xelatex_path=xelatex_path,
            lualatex_path=lualatex_path,
            error_message=error_message
        )
    
    @staticmethod
    def check_system_dependencies() -> SystemDependencies:
        """Check all system dependencies required for optimal manim operation"""
        latex_status = DependencyChecker.check_latex_installation()
        cairo_available, _ = DependencyChecker.check_command_available("pkg-config")
        pkg_config_available, _ = DependencyChecker.check_command_available("pkg-config")
        ffmpeg_available, _ = DependencyChecker.check_command_available("ffmpeg")
        manim_available, _ = DependencyChecker.check_command_available("manim")
        
        missing_packages = []
        if not latex_status.latex_available:
            missing_packages.append("LaTeX")
        if not cairo_available:
            missing_packages.append("Cairo")
        if not pkg_config_available:
            missing_packages.append("pkg-config")
        if not ffmpeg_available:
            missing_packages.append("FFmpeg")
        if not manim_available:
            missing_packages.append("Manim")
        
        return SystemDependencies(
            latex=latex_status.latex_available,
            cairo=cairo_available,
            pkg_config=pkg_config_available,
            ffmpeg=ffmpeg_available,
            manim=manim_available,
            missing_packages=missing_packages
        )
    
    @staticmethod
    def get_installation_instructions(missing_deps: List[str]) -> str:
        """Generate platform-specific installation instructions for missing dependencies"""
        if not missing_deps:
            return "All dependencies are installed and available."
        
        system = platform.system().lower()
        instructions = [f"Missing dependencies detected: {', '.join(missing_deps)}\n"]
        
        if system == "darwin":  # macOS
            instructions.append("Installation instructions for macOS:")
            if "LaTeX" in missing_deps:
                instructions.append("• LaTeX: brew install --cask mactex")
            if "Cairo" in missing_deps:
                instructions.append("• Cairo: brew install cairo")
            if "pkg-config" in missing_deps:
                instructions.append("• pkg-config: brew install pkg-config")
            if "FFmpeg" in missing_deps:
                instructions.append("• FFmpeg: brew install ffmpeg")
            if "Manim" in missing_deps:
                instructions.append("• Manim: pip install manim")
                
        elif system == "linux":
            instructions.append("Installation instructions for Linux (Ubuntu/Debian):")
            if "LaTeX" in missing_deps:
                instructions.append("• LaTeX: sudo apt-get install texlive-latex-base texlive-fonts-recommended texlive-latex-extra")
            if "Cairo" in missing_deps:
                instructions.append("• Cairo: sudo apt-get install libcairo2-dev")
            if "pkg-config" in missing_deps:
                instructions.append("• pkg-config: sudo apt-get install pkg-config")
            if "FFmpeg" in missing_deps:
                instructions.append("• FFmpeg: sudo apt-get install ffmpeg")
            if "Manim" in missing_deps:
                instructions.append("• Manim: pip install manim")
                
        elif system == "windows":
            instructions.append("Installation instructions for Windows:")
            if "LaTeX" in missing_deps:
                instructions.append("• LaTeX: Download and install MiKTeX from https://miktex.org/")
            if "Cairo" in missing_deps:
                instructions.append("• Cairo: Install via conda: conda install cairo")
            if "pkg-config" in missing_deps:
                instructions.append("• pkg-config: Install via conda: conda install pkg-config")
            if "FFmpeg" in missing_deps:
                instructions.append("• FFmpeg: Download from https://ffmpeg.org/ or use conda: conda install ffmpeg")
            if "Manim" in missing_deps:
                instructions.append("• Manim: pip install manim")
        else:
            instructions.append("Please refer to the official documentation for installation instructions on your platform.")
        
        return "\n".join(instructions)
    
    @staticmethod
    def generate_system_report() -> str:
        """Generate a comprehensive system dependency report"""
        deps = DependencyChecker.check_system_dependencies()
        latex_status = DependencyChecker.check_latex_installation()
        
        report = ["=== System Dependency Report ===\n"]
        report.append(f"Platform: {platform.system()} {platform.release()}")
        report.append(f"Python: {sys.version.split()[0]}\n")
        
        report.append("Dependency Status:")
        report.append(f"• LaTeX: {'✓' if deps.latex else '✗'}")
        if deps.latex and latex_status.pdflatex_path:
            report.append(f"  - pdflatex: {latex_status.pdflatex_path}")
        if deps.latex and latex_status.xelatex_path:
            report.append(f"  - xelatex: {latex_status.xelatex_path}")
        if deps.latex and latex_status.lualatex_path:
            report.append(f"  - lualatex: {latex_status.lualatex_path}")
            
        report.append(f"• Cairo: {'✓' if deps.cairo else '✗'}")
        report.append(f"• pkg-config: {'✓' if deps.pkg_config else '✗'}")
        report.append(f"• FFmpeg: {'✓' if deps.ffmpeg else '✗'}")
        report.append(f"• Manim: {'✓' if deps.manim else '✗'}")
        
        if deps.missing_packages:
            report.append(f"\n{DependencyChecker.get_installation_instructions(deps.missing_packages)}")
        else:
            report.append("\n✓ All dependencies are available for optimal rendering!")
        
        return "\n".join(report)

# MCP server
mcp = FastMCP()

# Get Manim executable path from environment variables or assume it's in the system PATH
MANIM_EXECUTABLE = os.getenv("MANIM_EXECUTABLE", "manim")   

# Manim output directory
TEMP_DIRS = {}
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(BASE_DIR, exist_ok=True)  

@mcp.tool()
def execute_manim_code(manim_code: str) -> str:
    """Execute the Manim code and return the path to the generated MP4 file"""
    tmpdir = os.path.join(BASE_DIR, "manim_tmp")  
    os.makedirs(tmpdir, exist_ok=True)  # Ensure the temp folder exists
    script_path = os.path.join(tmpdir, "scene.py")
    
    try:
        # Write the Manim script to the temp directory
        with open(script_path, "w") as script_file:
            script_file.write(manim_code)
        
        # Execute Manim with the correct path (remove -p flag to avoid auto-opening)
        result = subprocess.run(
            [MANIM_EXECUTABLE, script_path], 
            capture_output=True,
            text=True,
            cwd=tmpdir
        )

        if result.returncode == 0:
            TEMP_DIRS[tmpdir] = True
            
            # Find the generated MP4 file
            mp4_file = find_generated_mp4(tmpdir)
            
            if mp4_file:
                return f"Execution successful. Video generated at: {mp4_file}"
            else:
                return f"Execution successful but MP4 file not found. Check directory: {tmpdir}"
        else:
            # Enhanced error handling for LaTeX failures
            error_message = _handle_manim_error(result.stderr, result.stdout, manim_code)
            return error_message

    except Exception as e:
        return f"Error during execution: {str(e)}"


def find_generated_mp4(base_dir: str) -> Optional[str]:
    """Find the most recently generated MP4 file in the manim output directory"""
    try:
        # Manim typically creates videos in media/videos/scene/[quality]/ subdirectory
        media_dir = os.path.join(base_dir, "media", "videos", "scene")
        
        if not os.path.exists(media_dir):
            # Fallback: search entire base directory
            media_dir = base_dir
        
        mp4_files = []
        for root, dirs, files in os.walk(media_dir):
            for file in files:
                if file.endswith('.mp4'):
                    full_path = os.path.join(root, file)
                    mp4_files.append((full_path, os.path.getmtime(full_path)))
        
        if mp4_files:
            # Return the most recently created MP4 file
            latest_mp4 = max(mp4_files, key=lambda x: x[1])
            return latest_mp4[0]
        
        return None
    except Exception as e:
        print(f"Error finding MP4 file: {e}")
        return None



@mcp.tool()
def save_video_to_path(source_path: str, destination_path: str) -> str:
    """Copy the generated MP4 video to a specified destination path"""
    try:
        if not os.path.exists(source_path):
            return f"Source video file not found: {source_path}"
        
        # Ensure destination directory exists
        dest_dir = os.path.dirname(destination_path)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        
        # Copy the file
        shutil.copy2(source_path, destination_path)
        
        # Verify the copy was successful
        if os.path.exists(destination_path):
            file_size = os.path.getsize(destination_path)
            return f"Video successfully saved to: {destination_path} (Size: {file_size} bytes)"
        else:
            return f"Failed to save video to: {destination_path}"
            
    except Exception as e:
        return f"Error saving video: {str(e)}"


@mcp.tool()
def get_latest_video_path() -> str:
    """Get the path to the most recently generated video file"""
    try:
        tmpdir = os.path.join(BASE_DIR, "manim_tmp")
        if not os.path.exists(tmpdir):
            return "No manim temporary directory found. Generate a video first."
        
        mp4_file = find_generated_mp4(tmpdir)
        if mp4_file:
            file_size = os.path.getsize(mp4_file)
            return f"Latest video: {mp4_file} (Size: {file_size} bytes)"
        else:
            return "No MP4 files found in the manim output directory."
            
    except Exception as e:
        return f"Error finding latest video: {str(e)}"


@mcp.tool()
def cleanup_manim_temp_dir(directory: str) -> str:
    """Clean up the specified Manim temporary directory after execution."""
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            return f"Cleanup successful for directory: {directory}"
        else:
            return f"Directory not found: {directory}"
    except Exception as e:
        return f"Failed to clean up directory: {directory}. Error: {str(e)}"


@mcp.tool()
def check_rendering_capabilities() -> str:
    """Check available rendering engines and system dependencies for LaTeX and manim"""
    try:
        return DependencyChecker.generate_system_report()
    except Exception as e:
        return f"Error checking dependencies: {str(e)}"


@mcp.tool()
def check_latex_installation() -> str:
    """Check if LaTeX is installed and accessible for mathematical rendering"""
    try:
        latex_status = DependencyChecker.check_latex_installation()
        
        if latex_status.latex_available:
            available_engines = []
            if latex_status.pdflatex_path:
                available_engines.append(f"pdflatex: {latex_status.pdflatex_path}")
            if latex_status.xelatex_path:
                available_engines.append(f"xelatex: {latex_status.xelatex_path}")
            if latex_status.lualatex_path:
                available_engines.append(f"lualatex: {latex_status.lualatex_path}")
            
            return f"✓ LaTeX is available!\n\nAvailable engines:\n" + "\n".join(f"• {engine}" for engine in available_engines)
        else:
            missing_deps = ["LaTeX"]
            instructions = DependencyChecker.get_installation_instructions(missing_deps)
            return f"✗ LaTeX not found.\n\n{instructions}"
            
    except Exception as e:
        return f"Error checking LaTeX installation: {str(e)}"


@mcp.tool()
def generate_proper_axes_code(
    x_range: str = "-10,10,2", 
    y_range: str = "-6,6,2", 
    formula: str = "", 
    formula_position: str = "top_left"
) -> str:
    """Generate Manim code with properly configured coordinate axes that have correct scale and visible markings
    
    Args:
        x_range: X-axis range as "min,max,step" (e.g., "-10,10,2")
        y_range: Y-axis range as "min,max,step" (e.g., "-6,6,2") 
        formula: Optional mathematical formula to display
        formula_position: Position for formula (top_left, top_right, etc.)
    
    Returns:
        Complete Manim code with properly configured axes and optional positioned formula
    """
    
    # Parse ranges
    try:
        x_min, x_max, x_step = map(float, x_range.split(','))
        y_min, y_max, y_step = map(float, y_range.split(','))
    except ValueError:
        return "Error: Invalid range format. Use 'min,max,step' format (e.g., '-10,10,2')"
    
    # Position mappings
    position_coords = {
        "top_left": "UP * 2.5 + LEFT * 5",
        "top_right": "UP * 2.5 + RIGHT * 5", 
        "bottom_left": "DOWN * 2.5 + LEFT * 5",
        "bottom_right": "DOWN * 2.5 + RIGHT * 5",
        "top_center": "UP * 3",
        "bottom_center": "DOWN * 3"
    }
    
    pos_coord = position_coords.get(formula_position, position_coords["top_left"])
    
    # Generate tick marks for axes
    x_ticks = list(range(int(x_min), int(x_max) + 1, int(x_step)))
    y_ticks = list(range(int(y_min), int(y_max) + 1, int(y_step)))
    
    # Formula code if provided
    formula_code = ""
    if formula:
        formula_code = f'''
        # Mathematical formula positioned to avoid axis overlap
        formula = MathTex(r"{formula}")
        formula.move_to({pos_coord})
        
        # Add background for better visibility
        formula_bg = SurroundingRectangle(
            formula, 
            color=WHITE, 
            fill_color=BLACK, 
            fill_opacity=0.8,
            buff=0.1
        )
        
        self.add(formula_bg, formula)'''
    
    manim_code = f'''
from manim import *
import numpy as np

class ProperAxesScene(Scene):
    def construct(self):
        # Create coordinate axes with proper scale and markings
        axes = Axes(
            x_range=[{x_min}, {x_max}, {x_step}],
            y_range=[{y_min}, {y_max}, {y_step}],
            x_length=10,
            y_length=6,
            axis_config={{
                "color": BLUE,
                "stroke_width": 2,
                "include_numbers": True,
                "font_size": 24,
            }},
            x_axis_config={{
                "numbers_to_include": {x_ticks},
                "numbers_with_elongated_ticks": {x_ticks},
                "decimal_number_config": {{"num_decimal_places": 0}},
            }},
            y_axis_config={{
                "numbers_to_include": {y_ticks},
                "numbers_with_elongated_ticks": {y_ticks},
                "decimal_number_config": {{"num_decimal_places": 0}},
            }},
            tips=True,
        )
        
        # Create axis labels
        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("y")
        
        # Add axes and labels to scene
        self.add(axes, x_label, y_label)
        {formula_code}
        
        # Optional: Add grid lines for better readability
        grid = axes.get_coordinate_labels()
        
        self.wait(2)
'''
    
    return f"Generated Manim code with properly configured axes:\n\n{manim_code.strip()}"


@mcp.tool()
def get_axes_configuration_guide() -> str:
    """Get a comprehensive guide for configuring coordinate axes in Manim with proper scale and markings"""
    
    guide = """
=== Manim Coordinate Axes Configuration Guide ===

COMMON AXIS ISSUES & SOLUTIONS:

❌ PROBLEM: Missing axis markings/numbers
✅ SOLUTION: Set include_numbers=True in axis_config

❌ PROBLEM: Wrong scale or spacing
✅ SOLUTION: Properly configure x_range, y_range, and step values

❌ PROBLEM: Invisible or poorly positioned labels
✅ SOLUTION: Use get_x_axis_label() and get_y_axis_label()

PROPER AXES CONFIGURATION:

```python
axes = Axes(
    x_range=[-10, 10, 2],        # [min, max, step]
    y_range=[-6, 6, 2],          # [min, max, step]
    x_length=10,                 # Physical length on screen
    y_length=6,                  # Physical height on screen
    axis_config={
        "color": BLUE,
        "stroke_width": 2,
        "include_numbers": True,  # ← KEY: Shows numbers on axes
        "font_size": 24,
    },
    x_axis_config={
        "numbers_to_include": [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10],
        "numbers_with_elongated_ticks": [-10, -5, 0, 5, 10],
        "decimal_number_config": {"num_decimal_places": 0},
    },
    y_axis_config={
        "numbers_to_include": [-6, -4, -2, 0, 2, 4, 6],
        "numbers_with_elongated_ticks": [-6, -3, 0, 3, 6],
        "decimal_number_config": {"num_decimal_places": 0},
    },
    tips=True,  # Arrow tips on axes
)

# Add axis labels
x_label = axes.get_x_axis_label("x")
y_label = axes.get_y_axis_label("y")
```

KEY PARAMETERS EXPLAINED:

• x_range/y_range: [minimum, maximum, step_size]
• x_length/y_length: Physical size on screen (in Manim units)
• include_numbers: Shows numerical labels on axes
• numbers_to_include: Which specific numbers to show
• numbers_with_elongated_ticks: Which numbers get longer tick marks
• decimal_number_config: Controls decimal places shown
• tips: Whether to show arrow tips on axes

COMMON RANGES:
• Standard: x_range=[-10, 10, 2], y_range=[-6, 6, 2]
• Zoomed in: x_range=[-5, 5, 1], y_range=[-3, 3, 1]
• Large scale: x_range=[-20, 20, 5], y_range=[-12, 12, 3]

POSITIONING FORMULAS WITH AXES:
• Top-left: UP * 2.5 + LEFT * 5 (recommended)
• Top-right: UP * 2.5 + RIGHT * 5
• Bottom-right: DOWN * 2.5 + RIGHT * 5

Use generate_proper_axes_code() to automatically generate properly configured axes!
"""
    
    return guide


if __name__ == "__main__":
    mcp.run(transport="stdio")




