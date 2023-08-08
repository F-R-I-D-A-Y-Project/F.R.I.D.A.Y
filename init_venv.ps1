function InitVenv {
    python -m venv env
    & "env\Scripts\Activate.ps1"
    pip install -r requirements.txt
}

function friday {
    python aux.py $($args -join ' ')
}
