from routes.main import mainRoute
from routes.about import aboutRoute
from routes.eda import edaRoute

routepaths = {'Main' : mainRoute, #Landing Page
              'EDA' : edaRoute,
              'About' : aboutRoute,
              # Añadir mas rutas
              }