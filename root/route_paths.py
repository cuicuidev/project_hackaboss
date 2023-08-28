from routes.main import mainRoute
from routes.about import aboutRoute
from routes.eda import edaRoute
from routes.workflow import workflowRoute
from routes.model import modelRoute

routepaths = {'Main' : mainRoute, # Landing Page: explicamos el prjecto por encima y el stack utilizado
              'EDA' : edaRoute, # Hablamos del dataset
              'Workflow' : workflowRoute, # Hablamos del proceso de preprocesamiento de datos y entrenamiento del modelo, de la elección de stack y de la toma de decisiones, etc.
              'Model' : modelRoute, # Permitimos al usuario interactuar con el modelo final
              'About' : aboutRoute,
              # Añadir mas rutas
              }