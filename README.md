## NOTETIFY
Use virtual env for packages

# Crate venv
### NOTE: Python version <= 3.12 for this project to work

```python -m venv env```

# Run venv

```source env/Scripts/activate```
<!-- ```deactivate``` to turn off -->

# BEFORE you run this make sure that you are in a venv for this project
Run requirements.txt for packages

```pip install -r requirements.txt```

# Running django website
Make sure you are inside the django project folder ./notetify_project

```cd ./notetify_project```

```python manage.py runserver```


# Celery task in the background
Make sure you have redis-server running on your device/server.
Either redis-server for windows or on WSL.

Run celery task worker with command

```celery -A notetify_project worker --loglevel=info -P gevent```

# Edit tailwind css
Run these two commands together to see the changes updated

```python manage.py tailwind start```

```python manage.py runserver```


# Set the NPM branch for the django project (different based on installation)

```NPM_BIN_PATH = r"C:/Program Files/nodejs/npm.cmd"```


# Training
To run training run 

```python training/cnn_training.py```

## DEBUG_MODE
export DEBUG_MODE = True (Default False)
$DEBUG_MODE (Check current value)