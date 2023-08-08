function InitVenv {

    # Create a Python virtual environment
    python -m venv env

    # Activate the virtual environment
    & "env\Scripts\Activate.ps1"

    # Install requirements using pip
    pip install -r requirements.txt
}

function friday {

}
