## NOTETIFY
Use virtual env for packages

# Crate venv
### NOTE: Python version <= 3.12 for this project to work
```python -m venv env```

# Run venv
```source env/Scripts/activate```
<!-- ```deactivate``` to turn off -->

# BEFORE you run this make sure that you are in a venv for this project
# Run requirments.txt for packages
```pip install -r requirements.txt```

# Running django website
## Make sure you are inside the django project folder ./notetify_project
```cd ./notetify_project```
```python manage.py runserver```

# Edit tailwind css
Run these two commands together to see the changes updated
```python manage.py tailwind start```
```python manage.py runserver```

# Training
To run training run ```python training/cnn_training.py```

## DEBUG_MODE
export DEBUG_MODE = True (Default False)
$DEBUG_MODE (Check current value)