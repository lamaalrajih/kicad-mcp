@echo off
REM KiCAD-MCP Organization Script - Move tests to tests/ folder
echo Organizing test files into tests/ directory...

REM Create tests directory if it doesn't exist
if not exist tests mkdir tests
if not exist tests\outputs mkdir tests\outputs

REM Move test scripts to tests/
move test_*.py tests\ 2>nul
move debug_parser_output.py tests\ 2>nul
move extract_footprint_sample.py tests\ 2>nul
move find_q200_pads.py tests\ 2>nul

REM Move test outputs to tests/outputs/
move *_dsl.txt tests\outputs\ 2>nul
move schematic_index.txt tests\outputs\ 2>nul
move test_output_*.txt tests\outputs\ 2>nul

REM Clean up temporary files
del /Q cleanup.bat 2>nul
del /Q cleanup_plan.md 2>nul

echo Organization complete!
echo.
echo Directory structure:
echo   tests/               - Test scripts
echo   tests/outputs/       - Test output files
echo.
git status -s
