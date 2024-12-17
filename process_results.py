# imports
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import ortho_group
from scipy.stats import linregress
from scipy import linalg as la
from torch import nn
import torch
import os
from matplotlib.lines import Line2D
from scipy.stats import sem
from mpl_toolkits import mplot3d
import matplotlib
import logging
from time import time
import argparse

def Llayers(L,d=20,width=1000):
    """
    model class. Construct L-1 linear layers; bias terms only on last linear layer and final relu layer.
    """
    if L < 2:
        raise ValueError("L must be at least 2")
    if L == 2:
        linear_layers = [nn.Linear(d,width,bias=True)]
    if L > 2:
        linear_layers = [nn.Linear(d,width,bias=False)]
        if args.architecture == "relus" or args.architecture == "middlelinear": 
            linear_layers.append(nn.ReLU())
        for l in range(L-3):
            linear_layers.append(nn.Linear(width,width,bias=False))
            if args.architecture == "relus":
                linear_layers.append(nn.ReLU())
        linear_layers.append(nn.Linear(width,width,bias=True))

    relu = nn.ReLU()

    last_layer = nn.Linear(width,1)

    layers = linear_layers + [relu,last_layer]

    return nn.Sequential(*layers)

def gen_data(device,datasetsize,r,seed,std,labelnoiseseed,trainsize=2**18,testsize=2**10,d=20,funcseed=42,verbose=False,ood=False):

    ##Generate data with a true central subspaces of varying dimensions
    #generate X values for training and test sets
    np.random.seed(seed) #set seed for data generation
    trainX = np.random.rand(d,trainsize).astype(np.float32)[:,:datasetsize] - 0.5 #distributed as U[-1/2, 1/2]
    testX = np.random.rand(d,testsize).astype(np.float32) - 0.5 #distributed as U[-1/2, 1/2]
    #out of distribution datagen
    if ood:
        trainX *= 2 #now distributed as U[-1, 1]
        testX *= 2 #now distributed as U[-1, 1]
    ##for each $r$ value create and store data-gen functions and $y$ evaluations
    #geneate params for functions
    k = d+1
    if std == int(std):
        std = int(std)
    U = np.load(args.path + args.job_name+f"_labelnoise{std}/r{r}U.npy")
    Sigma = np.load(args.path + args.job_name+f"_labelnoise{std}/r{r}Sigma.npy")
    V = np.load(args.path + args.job_name+f"_labelnoise{std}/r{r}V.npy")
    A = np.load(args.path + args.job_name+f"_labelnoise{std}/r{r}A.npy")
    B = np.load(args.path + args.job_name+f"_labelnoise{std}/r{r}B.npy")
    if args.target == "specialized":
        logging.info("SPECIALIZED TARGET")
        Wprime = np.load(args.path + args.job_name+f"_labelnoise{std}/r{r}Wprime.npy")
        Bprime = np.load(args.path + args.job_name+f"_labelnoise{std}/r{r}Bprime.npy")
    #create functions
    np.random.seed(labelnoiseseed) #set seed for data generation
    if args.target == "specialized":
        def xprime(x):
            Wprimex = Wprime@x
            return np.maximum(0,Wprimex.T+Bprime).T
    def g(z): #active subspace function
        hidden_layer = (U*Sigma)@z
        hidden_layer = hidden_layer.T + B
        hidden_layer = np.maximum(0,hidden_layer).T
        return A@hidden_layer
    def f(x): #teacher network
        if args.target == "specialized":
            z = V.T@x    
        elif args.target == "SMIM":
            z = V.T@xprime(x)
        else:
            raise ValueError(f"{args.target} must be one of SMIM or specialized")
        eps = std*np.random.randn(x.shape[1])    
        return g(z) + eps
    logging.info(f(torch.zeros(d)))
    #generate data
    trainY = f(trainX).astype(np.float32)
    testY = f(testX).astype(np.float32)
    #move data to device
    logging.info(f"device: {device}")
    trainX = torch.from_numpy(trainX).T.to(device)
    trainY = torch.from_numpy(trainY).to(device)
    testX = torch.from_numpy(testX).T.to(device)
    testY = torch.from_numpy(testY).to(device)
    logging.info(f"trainX shape = {trainX.shape} trainY shape = {trainY.shape}")
    return trainX,trainY,testX,testY

def gen_validation(rs,labelnoise,validationsize = 2048):
    validationY = {}
    for r in rs:
        for k,std in enumerate(labelnoise):
            labelnoiseseed = 686 + k
            datagenseed = 1107
            logging.info(f"validation size = {validationsize} r = {r} label noise std = {std} label noise seed = {labelnoiseseed}")
            validationX,validationY[r,std] = gen_data(device,datasetsize=validationsize,r=r,seed=datagenseed,std=std,labelnoiseseed=labelnoiseseed)[:2]
    logging.info(f"min/max {(validationX.min(),validationX.max())}")
    return validationX,validationY

def gen_generalization(rs,labelnoise,generalizationsize = 2048):
    generalizationY = {}
    for r in rs:
        for k,std in enumerate(labelnoise):
            labelnoiseseed = 743 + k
            datagenseed = 555
            logging.info(f"generalization size = {generalizationsize} r = {r} label noise std = {std} label noise seed = {labelnoiseseed}")
            generalizationX,generalizationY[r,std] = gen_data(device,datasetsize=generalizationsize,r=r,seed=datagenseed,std=std,labelnoiseseed=labelnoiseseed)[:2]
    logging.info(f"min/max {generalizationX.min(),generalizationX.max()}")
    return generalizationX,generalizationY

def gen_ood(rs,labelnoise,oodsize = 2048):
    oodY = {}
    for r in rs:
        for k,std in enumerate(labelnoise):
            labelnoiseseed = 235 + k
            datagenseed = 333
            logging.info(f"ood size = {oodsize} r = {r} label noise std = {std} label noise seed = {labelnoiseseed}")
            oodX,oodY[r,std] = gen_data(device,datasetsize=oodsize,r=r,seed=datagenseed,std=std,labelnoiseseed=labelnoiseseed,ood=True)[:2]
    logging.info(f"min/max {oodX.min(),oodX.max()}")
    return oodX,oodY

def check_function(r,ns,verbose=False):
    # Check that most or all ReLU hyperplanes intersect the support of the distributions of the tests
    U = np.load(args.path + args.job_name+f"_labelnoise0/r{r}U.npy")
    Sigma = np.load(args.path + args.job_name+f"_labelnoise0/r{r}Sigma.npy")
    V = np.load(args.path + args.job_name+f"_labelnoise0/r{r}V.npy")
    A = np.load(args.path + args.job_name+f"_labelnoise0/r{r}A.npy")
    B = np.load(args.path + args.job_name+f"_labelnoise0/r{r}B.npy")
    W = (U*Sigma)@V.T
    rowwise1norms = np.linalg.norm(W,axis=1,ord=1)
    ratios = np.abs(B) / np.linalg.norm(W,axis=1,ord=1)
    rowwise2norms = np.linalg.norm(W,axis=1,ord=2)
    units = pd.DataFrame({"R2-cost contribution":np.abs(A)*rowwise2norms,"|b| / ||w||_1":ratios})

    datasetsize = 2048
    trainX = gen_data(device,datasetsize=datasetsize,r=r,seed=1,std=0,labelnoiseseed=0)[0]
    with torch.no_grad():
        for n in ns:
            units[f"# training active,n={n}"] = ((W@trainX[:n].cpu().numpy().T).T + B > 0).sum(axis=0)
            units[f"% training active,n={n}"] = units[f"# training active,n={n}"] / n
        units["# validation active"] = ((W@validationX.cpu().numpy().T).T + B > 0).sum(axis=0)
        units["% validation active"] = units["# validation active"] / datasetsize
        units["# generalization active"] = ((W@generalizationX.cpu().numpy().T).T + B > 0).sum(axis=0)
        units["% generalization active"] = units["# generalization active"] / datasetsize
        units["# ood active"] = ((W@oodX.cpu().numpy().T).T + B > 0).sum(axis=0)
        units["% ood active"] = units["# ood active"] / datasetsize

    logging.info(f"\nTOTALS:\n~~~~~~~\n {units.sum()}")
    logging.info(f"\nunit-wise table:\n~~~~~~~\n")
    logging.info(units)
    shift = -0.2
    width = 0.2
    plt.clf()
    for n in ns:
        if n >= np.max(ns)/2:
            plt.bar(units.index+shift,units[f"% training active,n={n}"],label=f"train,$n={n}$",width=width,tick_label=units["R2-cost contribution"].round(1))
            shift += width
    plt.bar(units.index+shift,units["% ood active"],label="ood",width=width)
    plt.ylim(0,1)
    plt.xticks(rotation=45)
    plt.title(f"How many samples is each unit active on? $r =$ {r}")
    plt.xlabel("ReLU Unit, labeled by R2-cost contribution")
    plt.ylabel("Proportion of samples")
    plt.axhline(0.5,linestyle=":",color="k")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"samples_active_by_unit_r{r}")

def process_job(r,n,L,wd,sigma,
                validationX,validationY,
                generalizationX,generalizationY,
                oodX,oodY,
                trainMSE_threshold=1e-2,
                verbose=False):
    res = {
        "r": r,
        "sigma":sigma,
        "n":n,
        "L":L,
        "lambda":wd
    } #results for a single job; one row in the later "res" dataframe
    
    logging.info(f"read in the files")
    if sigma == int(sigma):
        sigma = int(sigma)   
    paramname = args.path+args.job_name + f"_labelnoise{sigma}/N{n}_L{L}_r{r}_wd{wd}_epochs{args.epochs}"
    if os.path.exists(paramname+"testMSE.npy"):
        res["Test MSE"] = np.load(paramname+"testMSE.npy",allow_pickle=True).item()
        res["Train MSE"] = np.load(paramname+"trainMSEs.npy",allow_pickle=True)
        res["Weight Decay"] = np.load(paramname+"weightdecays.npy",allow_pickle=True)
        res["Learning Rate"] = np.load(paramname+"learningrates.npy",allow_pickle=True)

        model = Llayers(L,width=1000)
        model.to(device)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(paramname+"model.pt"))
        else:
            model.load_state_dict(torch.load(paramname+"model.pt"),map_location=torch.device('cpu'))
        model.eval()
    else:
        raise ValueError(f"{paramname+'testMSE.npy'} not found")

    res["Activations"] = args.architecture
    res["Final Train MSE"] = res["Train MSE"][-1]
    res["Final Weight Decay"] = res["Weight Decay"][-1]
    
    logging.info(f"compute MSEs")
    with torch.no_grad():
        for title,dataX,dataY in zip(["Validation", "In-Distribution", "Out-of-Distribution"],
                                     [validationX,  generalizationX,   oodX],
                                     [validationY,  generalizationY,   oodY]):
            predY = model(dataX)
            squared_err = (predY[:,0] - dataY[r,sigma])**2
            squared_err = squared_err.cpu().numpy()
            res[f"{title} Squared Errors"] = squared_err
            mse = nn.functional.mse_loss(predY[:,0],dataY[r,sigma]).item()
            res[f"{title} MSE"] = mse
            assert np.isclose(res[f"{title} MSE"],np.mean(squared_err))
            res[f"{title} SEM"] = sem(squared_err)
            res[f"{title} STD of Squared Errors"] = np.std(squared_err)
            if sigma > 0:
                res[f"{title} MSE$/\sigma^2$"] = mse/(sigma**2)
            else:
                res[f"{title} MSE$/\sigma^2$"] = np.nan

    ## evaluate gradients and compute singular values and active subspaces

    logging.info(f"compute ground truth active subspace")
    d = 20
    ln = sigma
    if int(ln) == ln:
        ln = int(ln)
    V = np.load(args.path + args.job_name+f"_labelnoise{ln}/r{r}V.npy")

    logging.info(f"evaluate gradients")
    generalizationX.requires_grad = True
    predY = model(generalizationX)
    grad = torch.autograd.grad(predY, generalizationX,
                            grad_outputs=torch.ones_like(predY),
                            create_graph=True)[0].detach().cpu().numpy()

    logging.info(f"compute active subspace and singular values")
    Uhat,Shat,VhatT = np.linalg.svd(grad)
    Vhat = VhatT.T[:,:r] #form the basis for the active subspace
    res["Gradient Evaluations"] = grad
    res["Gradient Singular Values"] = Shat
    res["Active Subspace"] = Vhat
    res["Active Subspace Distance"] = np.linalg.norm(V@V.T - Vhat@Vhat.T,2)
    res["Principal Angle (Degrees)"] = np.degrees(np.arcsin(res["Active Subspace Distance"]))

    return res

if __name__ == "__main__":
    #params of job to process
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ls",type=int, nargs="+", help = "tuple of L values")
    parser.add_argument("--rs",type=int, nargs="+", help = "tuple of r values")
    parser.add_argument("--ns",type=int, nargs="+", help = "tuple of n values")
    parser.add_argument("--wds",type=float, nargs="+", help = "tuple of wd values")
    parser.add_argument("--labelnoise",type=float, nargs="+", help = "tuple of labelnoise values")
    parser.add_argument("--epochs",type=int, help = "number of training epochs")
    parser.add_argument("--job_name",type=str, help = "name of job")
    parser.add_argument("--architecture",type=str, help = "network achitecture")
    parser.add_argument("--path",type=str, help = "path to output of job")
    parser.add_argument("--target",type=str, help = "type of target function used for training.")
    args = parser.parse_args()
    if not args.architecture in {"standard","relus","middlelinear"}:
        raise ValueError("architecture must be one of standard,relus,middlelinear")

    #set up logging
    logging.basicConfig(filename=f"log/process_{args.job_name}.out", encoding='utf-8', level=logging.INFO)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"device {device}")

    validationX,validationY = gen_validation(args.rs,args.labelnoise)
    generalizationX,generalizationY = gen_generalization(args.rs,args.labelnoise)
    oodX,oodY = gen_ood(args.rs,args.labelnoise)

    for r in args.rs:
        check_function(r,args.ns)

    job_results = []
    starttime = time()
    for r in args.rs:
        for n in args.ns:
            for L in args.Ls:
                for wd in args.wds:
                    for sigma in args.labelnoise:
                        logging.info(f"time {time()-starttime} r {r} n {n} L {L} wd {wd} sigma {sigma}")
                        res = process_job(r,n,L,wd,sigma,
                                            validationX,validationY,
                                            generalizationX,generalizationY,
                                            oodX,oodY)
                        job_results.append(res)
    res = pd.DataFrame(job_results)
    res.to_pickle(args.job_name+"_results")