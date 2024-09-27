# Docker/Singularity image definitions

Building on the `nvidia/cuda:11.8.0` docker image, we install the vulkan library and all the python dependencies for Octo and SimplerEnv.

Since these containers will be used for active development, the Octo and SimplerEnv packages should be mounted and installed.

Additionally, the python package `debugpy` is installed that allows vscode to connect a remote debug session inside the container.

## Examples

### Docker

```
docker run -v <Octo path>:/project/octo -v <SimplerEnv path>:/project/SimplerEnv -p 5678:5678 --gpus all --rm -it <image_name>
```
The port forwarding will be used by the debugger.

### Singulartiy

```
singularity shell --nv -B <Octo path>:/project/octo -B <SimplerEnv path>:/project/SimplerEnv <image_name>
```
There is no network isolation in Singularity, so there is no need to map any port.

### Note
 After mounting the code directories, they still have to be installed via:
```bash
pip install -e /project/octo
pip install -e /project/SimplerEnv/ManiSkill2_real2sim
pip install -e /project/SimplerEnv
``` 
Or you can automate this by using `singularity exec` and a bash file:
```bash
#!/bin/bash
pip install -e /project/octo
pip install -e /project/SimplerEnv/ManiSkill2_real2sim
pip install -e /project/SimplerEnv
python /project/octo/examples/02_finetune_new_observation_action.py --pretrained_path=hf://rail-berkeley/octo-small --data_dir=/project/aloha_sim_dataset --save_dir=/project/checkpoints
```

## Debugging
If you are trying to debug a file, use the `debugpy` package to start a session that vscode can connect to:

```
python -m debugpy --wait-for-client --listen 0.0.0.0:5678 script.py
```

Example for octo:
```
python -m debugpy --wait-for-client --listen 0.0.0.0:5678 examples/02_finetune_new_observation_action.py --pretrained_path=hf://rail-berkeley/octo-small --data_dir=...
```

Then in vscode, create a `launch.json` configuration for "Python Debugger: Remote Attach", e.g.
```json
{
    "configurations": [
        {
            "name": "Python Debugger: Remote Attach",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "."
                }
            ]
        },
    ]
}
```