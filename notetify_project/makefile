.PHONY: quick_start celery

start:
	python manage.py runserver

celery:
	celery -A notetify_project worker --loglevel=info -P gevent