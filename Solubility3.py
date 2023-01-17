#########################################
#### IMPORTING ESSENTIAL LIBRARIES ######
#########################################

import numpy as np
import pandas as pd
import streamlit as st 
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors


##########################################
#### DESIGNING THE PREDICITON MODEL ######
##########################################


st.title('ESOL-PREDICTION MODEL WEB APPLICATION')
st.header('Comprehensive ***Solubility (LogS)*** prediction-capable application based on affecting *parametres* of an ***agrochemical/Drug molecule***')
st.write('DISPLAYING THE DATASET FOR MWT, cLogP, RB, AP, LogS')
Link_for_the_discriptors='https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv'
dataset_of_molecules=pd.read_csv(Link_for_the_discriptors)
dataset_of_molecules
st.write('DISPLAYING THE ENTIRE DATASET EXCEPT THE LogS COLUMN')
X=dataset_of_molecules.drop(['logS'],axis=1)
X

# DISPLAYING THE DATASET FOR LogS (i.e.Y) #
st.write('DISPLAYING THE DATASET SINGULAR COLUMN FOR STORED LogS VALUES')
Y=dataset_of_molecules.iloc[:,-1]
Y

# CREATING A SUITABLE LINEAR REGRESSION MODEL #

from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score

model=linear_model.LinearRegression()
model.fit(X,Y)

Y_PRED=model.predict(X)
Y_PRED

# DESIGNING & DISPLAYING THE FORMULA #

print('Coefficients:',model.coef_)
print('Intercept:',model.intercept_)
print('Mean Squared Error(MSE):%.2f'%mean_squared_error(Y,Y_PRED))
print('Coefficient of Determination(R^2):%.2f'%r2_score(Y,Y_PRED))
print('LogS=%.2f%.2fLogP%.4fMW+%.4fRB%.2fAP'%(model.intercept_,model.coef_[0],model.coef_[1],model.coef_[2],model.coef_[3]))

# IMPORTING FILE AS .pkl #

import pickle
pickle.dump(model,open('Model.pkl','wb'))


###################################################
#### CALCULATING AND DISPLAYING THE PARAMETRES ####
###################################################

st.write('Calculated **Y-intercept values** are described above according to the Machine-learning model designed and pickled previously')
def AromaticProportion(m):
    aromatic_atoms=[m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
    aa_count=[]
    for i in aromatic_atoms:
     if i==True:
        aa_count.append(1)
    AromaticAtom=sum(aa_count)
    HeavyAtom=Descriptors.HeavyAtomCount(m)
    AR=AromaticAtom/HeavyAtom 
    return AR

def generate(smiles,verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData=np.arange(1,1)
    i=0
    for mol in moldata:

        desc_MolLogP= Descriptors.MolLogP(mol)
        desc_MolWt=Descriptors.MolWt(mol)
        desc_NumRotatableBonds=Descriptors.NumRotatableBonds(mol)
        desc_AromaticProportion=AromaticProportion(mol)

        row=np.array([desc_MolLogP,desc_MolWt,desc_NumRotatableBonds,desc_AromaticProportion])

        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData,row])
        i=i+1
    columnNames=["MolLogP","MolWt","NumRotatableBonds","AromaticProportion"]
    descriptors=pd.DataFrame(data=baseData,columns=columnNames)
    return descriptors

 # DESIGNING A TITLE #

st.write('''
# DISTRIBUTED PARAMETERIC ANALYSIS
''')

# DESIGNING THE IMPUT BAR #

st.sidebar.header('User Input Features')

# Reading The SMILES Input

SMILES_input="NCCC\nCCC\nCN"

SMILES=st.sidebar.text_area("SMILES Input", SMILES_input)
SMILES="C\n"+SMILES
SMILES=SMILES.split('\n')

st.header('Input SMILES')
SMILES[1:]

# Calculate Molecular Discriptors

st.header('COMPUTED VALUE FOR DESCRIPTORS')
Z=generate(SMILES)
Z[1:]


#######################################################
#### LOADING THE RESULTS BY USING PICKLING METHOD #####
#######################################################

load_model=pickle.load(open('Model.pkl','rb'))
prediction=load_model.predict(Z)

st.header('FINALISED PREDICTION OF LogS VALUE')
prediction[1:]