function [recMatrix] = LRLSHMDA_py(Wrr, Wdr, Wdd, lambdaM, lambdaD, lw)
    KM=Wrr;
    KD=Wdd;
    matDT=Wdr;
    [KD, KM] = GIPSim(matDT,1,1 );
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
    recMatrix=lw * FM' + (1 - lw) * FD;
end
