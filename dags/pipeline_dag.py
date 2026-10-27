"""Pipeline IRM : vérification du manifeste -> prétraitement -> entraînement -> évaluation -> sauvegarde.

Chaque tâche lance `python -m src.pipeline <étape>` dans l'environnement du projet (séparé d'Airflow).
Entraînement « léger » (tête seule, peu d'époques) ; évaluation sur la VALIDATION seulement (jamais le test).
Déclencher avec {"simulate_failure": true} pour provoquer un échec volontaire à la première tâche."""
from datetime import datetime, timedelta

try:                                   # Airflow 3
    from airflow.sdk import DAG
except ImportError:                    # Airflow 2
    from airflow import DAG
try:
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:
    from airflow.operators.bash import BashOperator

PY, PROJECT = "/opt/airflow/pipeline-venv/bin/python", "/opt/project"


def step(name, retries, **kw):
    return BashOperator(task_id=name, bash_command=f"cd {PROJECT} && {PY} -m src.pipeline {name}", append_env=True,
                        retries=retries, retry_delay=timedelta(seconds=30), execution_timeout=timedelta(hours=1), **kw)


with DAG(dag_id="pipeline_irm", start_date=datetime(2026, 1, 1), schedule=None, catchup=False,
         params={"simulate_failure": False}, tags=["irm", "projet"], doc_md=__doc__) as dag:
    # Une erreur de données ne se corrige pas en réessayant : 0 nouvel essai. Les autres étapes : 1 nouvel essai.
    validate = step("validate", 0, env={"PIPELINE_SIMULATE_FAILURE": "{{ 1 if params.simulate_failure else 0 }}"})
    preprocess, train, evaluate, save = (step(n, 1) for n in ("preprocess", "train", "evaluate", "save"))
    validate >> preprocess >> train >> evaluate >> save