function [ Gdat, f, e_r ] = estpsd( data, N, fs )
% This function is used to estimate the power spectral density from 
% time series information of one or more channels.

% Heather Clark
% July 12, 2012

%---- FUNCTION ARGUMENTS and OUTPUT -------------------------------
% data: an array with rows corresponding to samples in time, columns for channels
% N: number of points to use in the FFT
% fs: sampling frequency of data
%
% Gdat: final estimate of power spectral density, array with rows 
%       corresponding to frequency and columns for channels
% f: vector of frequencies that FFT has been evaluated at
% e_r: normalized random error in spectral output 
%------------------------------------------------------------------

if (floor(log2(N))~=log2(N))
    disp('Error: number of points for FFT should be a power of 2')
    return;
end
disp('PSD')
nchan = size(data,2);
Nsamp = size(data,1);
nrec = floor(Nsamp/N);

deltaf = fs/N;                   % Frequency bin size
nfreq = N/2 + 1;                 % Number of unique frequencies
f = (0:nfreq-1)*deltaf;          % Values of the frequency bin centres

Gdat = zeros(nfreq,nchan);

for rec=1:nrec
    ti = (rec-1)*N + 1;
    tf = ti + N - 1;

    d = data(ti:tf,:);
    D = fft(d);
    Sdd = D.*conj(D)/(N^2*deltaf);       % two-sided power spectral density
    Gdd = Sdd(1:nfreq,:);                % take unique frequency components, one-sided
    Gdd(2:end-1,:) = 2*Gdd(2:end-1,:);   % multiply by 2 except for DC and Nyquist

    Gdat = Gdat + Gdd;          
end
Gdat = Gdat/nrec;       % compute average over the records

% for chan=1:nchan
%     figure;
%     %semilogx(f,Gdat(:,chan));
%     loglog(f,Gdat(:,chan));
%     xlabel('Frequency (Hz)');
%     ylabel('PSD (Units^2/Hz)');
%     title(['Power spectral density for channel ' num2str(chan)]);
% end

% normalized random error of spectral calculation determined by number of 
% records used in averaging (Bendat and Piersol, section 8.5.4)
e_r = 1/sqrt(nrec);

end

