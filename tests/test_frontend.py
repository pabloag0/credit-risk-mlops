from streamlit.testing.v1 import AppTest


def test_frontend_arranca_sin_excepciones():
    """Verifica que la app de Streamlit carga sin lanzar ningún error."""
    at = AppTest.from_file("app/streamlit_app.py")
    at.run()
    assert not at.exception


def test_frontend_tiene_titulo():
    """Verifica que el título principal está presente."""
    at = AppTest.from_file("app/streamlit_app.py")
    at.run()
    assert any("Simulador" in t.value for t in at.title)


def test_frontend_tiene_formulario():
    """Verifica que los campos clave del formulario existen."""
    at = AppTest.from_file("app/streamlit_app.py")
    at.run()
    # Comprueba que hay campos numéricos (edad, ingresos, etc.)
    assert len(at.number_input) > 0
    # Comprueba que hay selectboxes (educación, propósito del préstamo, etc.)
    assert len(at.selectbox) > 0
