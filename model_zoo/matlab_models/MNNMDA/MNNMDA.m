function [M_recovery] = MNNMDA(Wrr, Wdr, Wdd, alpha, beta, tol1, tol2, maxiter, a, b)
    [dn,dr] = size(Wdr);
    [diseasesim, microbesim ] = GIPSim(Wdr,1,1 );
    %Wdd = diseasesim;
    %Wrr = microbesim;
    Wdd = (Wdd+diseasesim)/2.0;
    Wrr = (Wrr+microbesim)/2.0;
    T = [Wrr, Wdr'; Wdr, Wdd];
    [t1, t2] = size(T);
    trIndex = double(T ~= 0);
    [WW,iter] = DRMNN(alpha, beta, T, trIndex, tol1, tol2, maxiter, a, b);
    M_recovery = WW((t1-dn+1) : t1, 1 : dr);
end