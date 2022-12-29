function [recMatrix] = KATZHMDA_py(Wrr, Wdr, Wdd, beta)
    KM=Wrr;
    KD=Wdd;
    matDT=Wdr;
    [ KD,KM ] = GIPSim(matDT,1,1 );
    Sk2 = beta * matDT' + beta^2 * (KM*matDT' + matDT'* KD);
    recMatrix=Sk2';
end