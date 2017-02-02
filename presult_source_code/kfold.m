function varargout = kfold(fid, varargin) % If a function name is given, use it to call subfunction.
fid = strcat(fid,'_kfold');
fh = str2func(fid);
[varargout{1:nargout}] = fh(varargin{:});

function []=kfold(varargin)  % This was the original myFunc
MEEM_kfold(varargin);
SVM_kfold(varargin);
RF_kfold(varargin);
RT_kfold(varargin);
Logistic_kfold(varargin);

kfold('ME_kfold');
kfold('SVM_kfold');
kfold('RF_kfold');
kfold('RT_kfold');
kfold('Logistic_kfold');
end
end