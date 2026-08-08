from pathlib import Path
from streamlit.testing.v1 import AppTest

# Ruta absoluta a la app, independiente de desde dónde se ejecute pytest
APP_PATH = Path(__file__).parent.parent / "app" / "streamlit_app.py"


def test_frontend_arranca_sin_excepciones():
    """Verifica que la app de Streamlit carga sin lanzar ningún error."""
    at = AppTest.from_file(str(APP_PATH))
    at.run()
    assert not at.exception


def test_frontend_tiene_titulo():
    """Verifica que el título principal está presente."""
    at = AppTest.from_file(str(APP_PATH))
    at.run()
    assert any("Simulador" in t.value for t in at.title)


def test_frontend_tiene_formulario():
    """Verifica que los campos clave del formulario existen."""
    at = AppTest.from_file(str(APP_PATH))
    at.run()
    assert len(at.number_input) > 0
    assert len(at.selectbox) > 0
