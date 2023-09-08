from flask import Flask, render_template, request
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import numpy as np
import joblib
import io
from base64 import b64encode
from torch_geometric.nn.models import AttentiveFP
import torch
import utils


# loaded_model = joblib.load('../models/random_forest.joblib')
loaded_model = AttentiveFP(out_channels=4, # active or inactive
                    in_channels=4, edge_dim=2,
                    hidden_channels=200, num_layers=3, num_timesteps=2,
                    dropout=0.2)
loaded_model.load_state_dict(torch.load('./models/best_test_afp.model'))
print("model loaded succesfully")

grades_map = {0: ["D", "высокое", "> 100"], 
              1: ["C", "высокое", "10 - 100"], 
              2: ["B", "невысокое", "1 - 10"], 
              3: ["A", "невысокое", "< 1"]}

app = Flask(__name__)


def ECFP4(mol):
  return np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))


def MolImage(smiles):
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m is None:
        raise TypeError
    img = Draw.MolToImage(m, size=(480, 480))

    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format=img.format)
    imgByteArr.seek(0)
    imgByteArr = imgByteArr.getvalue()

    return imgByteArr


@app.route('/')
def index():
    return render_template('desktop-1.html')


@app.route('/aboutus')
def aboutus():
    return render_template('desktop-2.html')


@app.route('/app', methods=['GET', 'POST'])
def application():
    if request.method == 'POST':
        smi = request.form['text']
        print(smi)
        m = Chem.MolFromSmiles(smi)
        res = ""
        image = None
        if m is None:
            res = "Формула введена некорректно"
        else:
            data = utils.MyDataset([smi], [0])
            loader = utils.DataLoader(data)
            grade = grades_map[utils.predict(loader, loaded_model)[0]]
            res = f'Категория {grade[0]}, значение аффинности молекулы к CRBN находится \
                    в промежутке {grade[2]} µМ'

            image = 'data:image/png;base64,' + b64encode(MolImage(smi)).decode('ascii') 

        return render_template('desktop-3.html',res=res,smi=smi, image_data=image)
    else:
        return render_template('desktop-3.html')

if __name__ == '__main__':
    app.run()