@echo off
python generate_end_to_end_nb.py
jupyter nbconvert --to notebook --execute --inplace end_to_end_bug_severity_project.ipynb
pause
