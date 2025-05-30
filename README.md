# 🚀 Gensyn RL-Swarm RTX 5000 Series Fork

Pre-patched and tested fork of Gensyn RL-Swarm for NVIDIA RTX 5090/5080/5070 (Blackwell sm_120).

## ✅ What's Included
- All RTX 5000 series patches pre-applied
- Working dependency versions locked
- vLLM disabled (incompatible with sm_120)
- Flash Attention disabled
- DHT bootstrap fixes
- Tested for 30+ minutes on RTX 5090

## 🎯 Quick Install
```bash
git clone https://github.com/nuxxor/rl-swarm-rtx5000
cd rl-swarm-rtx5000
python -m venv gensyn_env
source gensyn_env/bin/activate
pip install -r requirements-rtx5000.txt
./run_rl_swarm.sh
