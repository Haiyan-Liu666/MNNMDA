clear all

beta = 0.01;
load .\dataset\HMDAD\interaction;
%interaction1=interaction;
%KM=textread('.\dataset\Disbiome\microbe_features.txt');
%KD=textread('.\dataset\Disbiome\disease_features.txt');
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
    [ KD,KM ] = GIPSim(matDT,1,1 );
    Sk2 = beta * matDT' + beta^2 * (KM*matDT' + matDT'* KD);
    recMatrix=Sk2';
    V=[Vn(vn);Vp(vp)];
    MatPredict(V)=recMatrix(V);
end
   [AUC,AUPR,Acc,Sen,Spe,Pre]=ROCcompute(MatPredict(),interaction(),1); 
   save('.\Results\MatPredict_KATZHMDA-Disbiome.mat','MatPredict')
   [AUC AUPR]
