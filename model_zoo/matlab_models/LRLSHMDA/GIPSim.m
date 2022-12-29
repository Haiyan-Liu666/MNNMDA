function [ kd,kl ] = GIPSim(interaction,gamadd,gamall)
% Xiang  2019-11-16 
%interaction: relation matrix between disease and miRNA,  column:miRNA  row:disease
[nd, nl] = size( interaction  ); 
%calculate gamad for Gaussian kernel calculation
    for i=1:nd
        sd(i)=norm(interaction(i,:))^2;
    end
    gamad=nd/sum(sd')*gamadd;
    
 %calculate gamal for Gaussian kernel calculation
    for i=1:nl
        sl(i)=norm(interaction(:,i))^2;
    end
    gamal=nl/sum(sl')*gamall;
    
    %calculate Gaussian kernel for the similarity between disease: kd
    for i=1:nd
        for j=1:nd
    kd(i,j)=exp(-gamad*(norm(interaction(i,:)-interaction(j,:)))^2);
        end
    end
    
    %calculate Gaussian kernel for the similarity between microbe: kl
        for i=1:nl
            for j=1:nl
                kl(i,j)=exp(-gamal*(norm(interaction(:,i)-interaction(:,j)))^2);
            end
        end 
