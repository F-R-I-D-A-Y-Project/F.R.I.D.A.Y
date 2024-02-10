# F.R.I.D.A.Y


Python project developed for the discipline "Laboratório de Programação 2", at "Instituto Militar de Engenharia"

## About 

F.R.I.D.A.Y is a chatbot and personal assistant AI developed in Python, capable of answering question made by user. The model is a fine-tuned version of the GPT2-large pretrained model, trained with the dataset "truthful_qa", by ########. The fine-tuning was made using QLoRA technique, to reduce the necessary computational power.

## Team members

- Fabricio Asfora Romero Assunção
- Roberto Suriel de Melo Rebouças
- Johannes Elias Joseph Salomão

## Compatibility

F.R.I.D.A.Y is compatible with Python 3.11+

## How to use
To create and activate the Python virtual environment, use:

- On Linux:
```shell
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
```
- On Windows:
```shell
    python -m venv env
    .\env\Scripts\activate
    pip install -r requirements.txt
```

Once the installation is complete, run:
```shell
python main.py
```

## About QLoRA

Quantized Low-Rank Adaptation (QLoRA) is a fine-tuning technique that proposes a way of reducing the computation power and data amount required to fine-tune an LLM. QLoRA offers several benefits in the NLP area, such as reducing significantly the training time and the memory usage while maintaining a good performance. This is how it works:

- Quantization: QLoRA reduces the precision of the weights in the pre-trained LLM from 32-bit to 4-bit (or even lower). This significantly reduces the memory footprint and allows for faster computations.
- Low-rank adapters: Instead of modifying all the parameters, QLoRA introduces a small set of learnable "adapters" that capture the specific adjustments needed for fine-tuning. These adapters have a "low-rank" structure, meaning they use fewer parameters than the original LLM, further improving efficiency.
- Fine-tuning: During fine-tuning, only the adapters are updated, not the entire LLM. This makes the process faster and requires less data compared to traditional fine-tuning.

[Check the original paper for more](https://arxiv.org/pdf/2305.14314v1.pdf)

## Future challenges

- Memory:F.R.I.D.A.Y. is still a new project, and as such, several features seen in proeminent AI chats, such as ChatGPT or Gemini, are still missing. One prime example is the bot's "memory" of a conversation. That is the result of the integration of the LLM with a vector database in a Chain of Thought. It will, in the future, be implemented in F.R.I.D.A.Y. as well. 
- Dataset: truthful_qa is an excellent dataset, but its too small for the purposes of F.R.I.D.A.Y. As such, a new dataset will be chosen in the future
- Pretrained LLM: Due to physical constraints, we weren't able to train an LLM better than GPT2-large. That won't be a problem in the future, and a better LLM will be selected.