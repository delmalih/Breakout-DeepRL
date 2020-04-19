# Snake-DeepRL

Course project of the Reinforcement Learning course taught by Dr. Matthieu J.
Team Members - [David El Malih](https://github.com/delmalih) and [Sai Deepesh Pokala](https://github.com/saideepesh)

![game](https://i.imgur.com/h3bJeXT.png)

### Requirements

* Python 3
* `pip3 install -r requirements.txt`

### How to use it ?

Before running the `run_agent.py` script, feel free to edit the `constants.py` file.

##### Run a random player :

```
python3 run_agent.py -a Random
```

##### Train a CNN agent :

```
python3 run_agent.py -a CNN --train
```

![training](https://i.imgur.com/OVaNTR6.png)

##### Run a CNN agent (with the pretrained model in `<OUTPUT_PATH>/model.pth`) :

```
python3 run_agent.py -a CNN
```
