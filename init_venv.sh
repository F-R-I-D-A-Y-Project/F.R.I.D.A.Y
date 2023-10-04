function activate_venv(){
    chmod +x runner.py 
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    deactivate
}

function friday(){
    source env/bin/activate
    python runner.py $@
    deactivate
}
