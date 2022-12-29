clear all
%load virusdrug;
%load disbiome_interaction;
load .\dataset\HMDAD\interaction;
%diseasesim=textread('.\dataset\Disbiome\disease_features.txt');
%microbesim=textread('.\dataset\Disbiome\microbe_features.txt');
maxiter = 300;
alpha = 1;
beta = 1;
tol1 = 2*1e-3;
tol2 = 1*1e-5;  
%[dn,dr] = size(interaction1);
[dn,dr] = size(interaction);
Vp=find(interaction()==1);
Vn=find(interaction()==0);
MatPredict=zeros(dn,dr);
Ip=crossvalind('Kfold',numel(Vp),5);
In=crossvalind('Kfold',numel(Vn),5);
for I=1:5
    vp=Ip==I;
    vn=In==I;
    matDT=interaction;
    matDT(Vp(vp))=0;
    [ diseasesim,microbesim ] = GIPSim(matDT,1,1 );
    T = [microbesim, matDT'; matDT, diseasesim];
    [t1, t2] = size(T);
    trIndex = double(T ~= 0);
    [WW,iter] = DRMNN(alpha, beta, T, trIndex, tol1, tol2, maxiter, 0, 1);
    recMatrix = WW((t1-dn+1) : t1, 1 : dr);
    V=[Vn(vn);Vp(vp)];
    MatPredict(V)=recMatrix(V);
end
   [AUC,AUPR,Acc,Sen,Spe,Pre]=ROCcompute(MatPredict(),interaction(),1); 
   save('.\Results\MatPredict_DRMNN-HMDAD.mat','MatPredict')
   [AUC AUPR]

 