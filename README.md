# To create a virtual env follow the next steps:

1. clone the repo
```
git clone https://github.com/TairCohen/personal-nutritionist-agent.git
cd personal-nutritionist-agent
```

2. create a virtual env and install relevant packages
```
python3 -m venv agent-env --upgrade-deps
source agent-env/bin/activate
pip install -r requirements.txt
```

3. create .env file and load keys to terminal by:
```
export $(grep -v '^#' .env | xargs)
```

4. install jupyter karnel on the virtual env
```
pip install ipykernel jupyter
python -m ipykernel install --user --name=agent-env --display-name "agent env"
```

5. To stop virtual env running, type:
```
deactivate agent-env 
 ```