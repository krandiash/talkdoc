# 🎙️ talkdoc

Point to a Github repo, and get an index that you can use for asking
questions about the code.

Built with 🚀 [Meerkat](https://github.com/hazyresearch/meerkat).

<video src="https://user-images.githubusercontent.com/10166085/225598887-808a5e03-c36f-43a3-912e-e829ec713608.mov" controls="controls" autoplay loop="loop" style="max-width: 400px;">
</video>

<video src="https://user-images.githubusercontent.com/10166085/225599063-7c0419cc-5b7a-4d02-8536-c205213b650d.mov" controls="controls" autoplay loop="loop" style="max-width: 400px;">
</video>

To get started,

```bash
git clone https://github.com/krandiash/talkdoc.git
conda create -n talkdoc python=3.9
pip install -r requirements.txt
```

Make sure to setup your OpenAI key and/or Cohere key in your environment.

```bash
export OPENAI_API_KEY=<your_key>
export COHERE_API_KEY=<your_key>
```

## 📇 Indexing
Plug in the name of any public Github repo!
```bash
python index-docs.py --repo <public_repo_url> --prompt-only
```
For example, we provide a demo for the `meerkat` (https://github.com/hazyresearch/meerkat) and `numpy` (https://github.com/numpy/numpy) repos.

```bash
python index-docs.py --repo https://github.com/hazyresearch/meerkat --prompt-only
# Type in a query like "How do I create an interactive visualization with a table and a scatterplot in Python with Meerkat?"
# Set n to a number like 5 to retrieve and put the top 5 most relevant results into the prompt
```
This will generate a prompt that you can stick into ChatGPT or GPT-4.

If you want ChatGPT to answer your questions programatically, just remove the `--prompt-only` flag.
```bash
python index-docs.py --repo https://github.com/hazyresearch/meerkat
```

Note that the `numpy` example is only supported right now with `--model cohere/small` (due to the size of the repo, I cheaped out).

### ⚙️ Changing indexing model
You can change the indexing model by changing the `--model` flag. Pass in any model from OpenAI with `openai/...` e.g. `openai/text-embedding-ada-002` or any model from Cohere with `cohere/...` e.g. `cohere/small`.

## 📝 Indexing in the Notebook
The same indexing workflow is available in the notebook. Just run the cells in `index-docs.ipynb`.

This might be more fun to use, since it shows off some of the cool features of Meerkat in visualizing and playing with the data!

For example, when a Table view pops up, try:
1. double clicking on the numbers on the left side to open up a modal view.
2. in the modal view, click on the column names on the left to go through the different columns.

The last cell shows off some of the cool interactive GUI stuff in Meerkat. For example,
we're popping up a button there for you to be able to copy the prompt over to ChatGPT / GPT-4.
