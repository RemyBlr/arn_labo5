# Run your image classifier on your phone

Once you have completed and trained the notebook, you can test your model live on your phone camera. No app to install — you just open a link in Safari or Chrome, on Android or iPhone.

There are **three paths** to choose from, depending on your machine. What differs between them is only **how you export your trained model**:

- **Path A — Mac or Linux:** train and export entirely on your laptop.
- **Path B — Windows:** train on your laptop as usual, then use Google Colab *only for the conversion step*.
- **Path C — Google Colab:** if you'd rather not set up anything locally, train and export entirely in Google Colab.

The reason Windows needs Colab: the TF.js conversion tool clashes with TensorFlow and NumPy versions on Windows in ways that are very hard to untangle (the `uvloop`, `tensorflow_hub`, and `protobuf` errors are all symptoms of this). Colab runs on Linux in your browser, where the conversion just works.

## Which classifier file do I use?

There are two versions of the web app. Use the one that matches how you exported:

- **`classifier_template_mac.html`** — for **Path A** (local export with `save_keras_model`).
- **`classifier_template_win.html`** — for **Path B and Path C** (any export that goes through Colab's `convert_tf_saved_model`).

They differ only in how they load the model internally; everything else is identical. All paths end at the same "run it on your phone" section at the end of this guide.

---

# Path A — Mac or Linux

## Step 1 — Set up the Python environment

You need Python 3.10 and TensorFlow 2.15. Open a terminal and run these commands one by one:

```bash
conda create -n Cours_ARN_2026 python=3.10 -y
conda activate Cours_ARN_2026

pip install tensorflow==2.15.0
pip install tensorflowjs
pip install tf-keras-vis matplotlib
pip install numpy pandas scikit-learn Pillow
pip install tflite-support
pip install ipykernel
python -m ipykernel install --user --name Cours_ARN_2026 --display-name "Cours_ARN_2026 (Python 3.10)"
```

Why TensorFlow 2.15 specifically? Because it uses Keras 2 by default, which is compatible with the conversion tool that produces the web model. Newer versions of TensorFlow use Keras 3 which breaks the conversion.

## Step 2 — Train your model and export it

Open the TransferLearning_part2.ipynb, make sure the kernel is set to `Cours_ARN_2026`, and run all cells from the top.

The last cell exports your trained model for the web app:

```python
import tensorflowjs as tfjs
import json

# Save the trained model to a Keras-2 .h5 file
model.save("model.h5")

# Convert to the TensorFlow.js web format -> creates a "tfjs_model/" folder
tfjs.converters.save_keras_model(model, 'tfjs_model')

with open('tfjs_model/labels.json', 'w') as f:
    json.dump(list(LABEL_NAMES), f)
```

After this cell runs you will see a `tfjs_model/` folder appear next to the notebook. It contains your model weights and a `labels.json` file with your class names. You may see a warning about HDF5 format — that is expected, ignore it. Rename `tfjs_model/` to `model/`.

`LABEL_NAMES` is set automatically from your dataset folder names earlier in the notebook — you don't need to change anything. Just make sure your dataset folders are named exactly how you want the labels to appear in the app (for example, `stylos`, `livres`, `enveloppes`).

## Step 3 — Use the Mac/Linux classifier

This path uses **`classifier_template_mac.html`**. Place it in the same folder as your `model/` folder, then jump to **"Run it on your phone"** at the end of this guide.

---

# Path B — Windows : train locally, convert in Colab

You keep training on your own machine in the usual way, and only borrow Colab for the one step that breaks on Windows — the conversion. This keeps your familiar local workflow and your own dataset folders.

## Step 1 — Train and export locally

Train the notebook on your machine as usual. At the end, instead of converting, just export the model as a SavedModel:

```python
model.export("exported_model")
```

This creates an `exported_model/` folder next to your notebook. You do **not** need tensorflowjs installed locally for this — `model.export` is plain TensorFlow.

## Step 2 — Put the exported model on Google Drive

Upload the whole `exported_model/` folder to your Google Drive, somewhere you can find it again — for example `MyDrive/ARN26/TP5/exported_model`. Nothing else needs to go to Drive, just that folder.

## Step 3 — Convert in Colab

Open a new notebook at [https://colab.research.google.com](https://colab.research.google.com) and run the following cells.

Install the conversion tool:

```python
%pip install tensorflowjs
```

Mount your Drive so Colab can read the exported model (it will ask you to authorise access):

```python
from google.colab import drive
drive.mount('/content/drive')
```

Convert the SavedModel to the TF.js format. Set `path_model` to where you put the folder on your Drive, and set `LABEL_NAMES` to **your own** class names — edit the example list below to match your project:

```python
import tensorflowjs as tfjs
import json

path_model = "drive/MyDrive/ARN26/TP5/exported_model" # Edit this to your own path

# Edit this list to your own classes. The sorted() wrapper guarantees the
# order matches the model's outputs, so you can list them in any order.
LABEL_NAMES = sorted(["stylos", "livres", "enveloppes"])

# Convert from SavedModel to TF.js format
tfjs.converters.convert_tf_saved_model(path_model, "tfjs_model")

with open('tfjs_model/labels.json', 'w') as f:
    json.dump(list(LABEL_NAMES), f)
```

## Step 4 — Download the model folder

```python
import shutil
from google.colab import files

Folder = 'tfjs_model'

# Zip the directory
shutil.make_archive(Folder, 'zip', Folder)

# Download the zipped file
files.download(f'{Folder}.zip')
```

A `tfjs_model.zip` will download to your computer. Unzip it and rename the resulting `tfjs_model/` folder to `model/`.

## Step 5 — Use the Windows classifier

This path produces a **graph model**, so it uses **`classifier_template_win.html`**. Place it next to your `model/` folder, then jump to **"Run it on your phone"** at the end of this guide.

---

# Path C — Windows (alternative): everything in Colab

If you'd rather not run anything locally at all, you can train and export entirely in Colab. You use the same provided notebook, just uploaded to Colab.

## Step 1 — Open the notebook in Colab

Go to [https://colab.research.google.com](https://colab.research.google.com), sign in with a Google account, and upload the provided notebook (File → Upload notebook).

In the top right, click **Connect** to start a runtime. The free CPU runtime is enough for this assignment.

## Step 2 — Install tensorflowjs at the top of the notebook

In the very first cell, add:

```python
%pip install tensorflowjs
```

Colab comes with TensorFlow pre-installed (a recent Keras 3 version), so you only need to add the conversion tool. The install takes about a minute.

## Step 3 — Upload your dataset

Drag your `dataset_train/` and `dataset_test/` folders into Colab's file panel on the left (the folder icon). Colab gives every session a temporary workspace, so the files live there only while the notebook is running.

Then run the notebook from the top. Training works exactly as it does locally.

## Step 4 — Export the model (Colab export cell)

Because Colab uses Keras 3, the export cell is different from the Mac/Linux one. Replace the last cell with this:

```python
import tensorflowjs as tfjs
import json

# Export as SavedModel (Keras 3 syntax)
model.export("saved_model_dir")

# Convert from SavedModel to TF.js format
tfjs.converters.convert_tf_saved_model("saved_model_dir", "tfjs_model")

with open('tfjs_model/labels.json', 'w') as f:
    json.dump(list(LABEL_NAMES), f)
```

After it runs you will see a `tfjs_model/` folder in Colab's file panel.

## Step 5 — Download the model folder

Colab files disappear when the session ends, so download the folder. Add this cell and run it:

```python
import shutil
from google.colab import files

Folder = 'tfjs_model'

# Zip the directory
shutil.make_archive(Folder, 'zip', Folder)

# Download the zipped file
files.download(f'{Folder}.zip')
```

A `tfjs_model.zip` will download to your computer. Unzip it and rename the resulting `tfjs_model/` folder to `model/`.

## Step 6 — Use the Windows classifier

This path produces a **graph model**, so it uses **`classifier_template_win.html`**. Place it next to your `model/` folder, then continue to **"Run it on your phone"** below.

---

# Run it on your phone

This part is the same for all three paths. You need the correct classifier HTML (mac or win, per the path you used) and your `model/` folder together in one directory.

## Step 1 — Set up ngrok (one time only)

Your phone needs to access your laptop over a secure connection. ngrok creates that connection for you in one command.

**Create a free account** at [https://ngrok.com](https://ngrok.com). No credit card needed.

Once logged in, copy your authtoken from the ngrok dashboard. Then in a terminal:

```bash
ngrok config add-authtoken YOUR_TOKEN_HERE
```

Replace `YOUR_TOKEN_HERE` with the token you copied. You only need to do this once.

> **Installing ngrok:** on Mac use `brew install ngrok`. On Windows, download the installer from [https://ngrok.com/download](https://ngrok.com/download).

## Step 2 — Serve the folder and open the tunnel

Make sure your classifier HTML and the `model/` folder are in the same directory. Open two terminals.

In the first, go to that folder and start a local web server:

```bash
cd path/to/your/PW_5/folder
python -m http.server 8000
```

(Use `python3` on Mac/Linux if `python` is not found.) Leave it running. In the second terminal:

```bash
ngrok http 8000
```

ngrok will print something like:

```
Forwarding   https://xxxx-xx-xx.ngrok-free.app -> http://localhost:8000
```

## Step 3 — Open it on your phone

On your phone, open Safari or Chrome and go to the ngrok address followed by your classifier filename:

```
https://xxxx-xx-xx.ngrok-free.app/classifier_template_mac.html
```

or, for the Windows paths:

```
https://xxxx-xx-xx.ngrok-free.app/classifier_template_win.html
```

The first time you open a ngrok link it shows a warning page — just click **Visit Site**. Then allow camera access when the browser asks. Your classifier is now running live.

---

# Retrained your model?

- **Path A:** re-run the last notebook cell, rename `tfjs_model/` to `model/` (replacing the old one), and refresh the page on your phone.
- **Path B:** re-export locally with `model.export("exported_model")`, re-upload it to Drive (replacing the old folder), re-run the Colab conversion and download cells, then swap in the new `model/` folder and refresh.
- **Path C:** re-run the Colab export and download cells, swap in the new `model/` folder, and refresh.
