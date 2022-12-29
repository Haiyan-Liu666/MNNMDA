clear all
lambdaM = 0.1;
lambdaD = 0.1;
lw = 0.1;
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
    sumM=sum(KM,2);
    sumD=sum(KD,2);
    DM=diag(sumM,0);
    DD=diag(sumD,0);
    DM_2=diag(sumM.^(-1/2),0);
    DD_2=diag(sumD.^(-1/2),0);
    LM=DM_2*(DM-KM)*DM_2;
    LD=DD_2*(DD-KD)*DD_2;
    FM = KM *pinv(KM + lambdaM * (LM*KM))*matDT';
    FD = KD *pinv(KD + lambdaD * (LD*KD))*matDT;
    F=lw * FM' + (1 - lw) * FD;
    V=[Vn(vn);Vp(vp)];
    MatPredict(V)=F(V);
end
   [AUC,AUPR,Acc,Sen,Spe,Pre]=ROCcompute(MatPredict(),interaction(),1); 
   save('.\Results\MatPredict_LRLSHMDA-HMDAD.mat','MatPredict')
   [AUC AUPR]
