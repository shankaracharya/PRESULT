function ME_train(varargin)
fprintf('\n Example command line: "./presult train ME --train_data <training_data.txt>" \n');
p = inputParser;
addParameter(p,'train_data','pid_train_data.txt',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'emiters','10');
addParameter(p,'actfunc','logistic',@ischar);
addParameter(p,'var','variables.txt',@ischar);
addParameter(p,'out','ME',@ischar);
addParameter(p,'hexp','8');
addParameter(p,'hgate','20');
addParameter(p,'nexp','5');
addParameter(p,'mstep_iters','7')
addParameter(p,'randm','shuffle',@ischar);
addParameter(p,'logistic_regression','on',@ischar);
addParameter(p,'p_val','0.01');
addParameter(p,'title','Mixture of Experts',@ischar);

new_array = varargin(:);
for i =1:size(new_array,1)
     for j = 1:size(p.Parameters,2)
         param=p.Parameters{j};
         array_entry=new_array{i};
         first_check=strcat('-',param);
         second_check=strcat('--',param);
         if strcmp(array_entry,first_check)
            new_array{i}=param;
         end
         if strcmp(array_entry,second_check)
            new_array{i}=param;
         end
     end
end

new_array(strcmp('yes',new_array))={'shuffle'};
new_array(strcmp('no',new_array))={'default'};

parse(p,new_array{:});

train_file = (p.Results.train_data);
var = (p.Results.var);
csv = (p.Results.csv);
miss = (p.Results.missing_value);
n = (p.Results.emiters);
mstep_iters = (p.Results.mstep_iters);
actfunc = (p.Results.actfunc);
out = (p.Results.out);
nexp = (p.Results.nexp);
hgate = (p.Results.hgate);
hexp = (p.Results.hexp);
ran = (p.Results.randm);
log_reg = (p.Results.logistic_regression);
p_val = (p.Results.p_val);
title_me = (p.Results.title);
nexp = str2double(nexp);
mstep_iters = str2double(mstep_iters);
n = str2double(n);
hgate = str2double(hgate);
hexp = str2double(hexp);
p_val = str2double(p_val);

if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', train_file, '--output', 'ME_normalized_train_data.txt','-var',var);

else
   perl('newest_normalization.pl', '--input', train_file, '--output', 'ME_normalized_train_data.txt', '--csv','-var',var);
end


q = load('ME_normalized_train_data.txt');

rownumbers = size(q,1);
%%%%%%%%%%%%%%%%%%%%
%DataSorting start
%%%%%%%%%%%%%%%%%%%%
rng(ran);
ran = rand(rownumbers,1);
data_file2 = [ran q];
data_file3 = sortrows(data_file2,1);
data_file = data_file3(:,2:end);
%%%%%%%%%%%%%%%%%%%%%
%Data Sorting end
%%%%%%%%%%%%%%%%%%%%%%
ninput = size(data_file,2);
x = data_file(:,1:ninput-1);
t = data_file(:,ninput);
net =  mixmlp(ninput-1,hexp,1,hgate,nexp,actfunc,'standard');
options  = zeros(1,18);
options(1)  =  -1;  % no display of the error value within the M-step 
options(14) =  mstep_iters;  % the number of iterations in the M-step   
[net,~] = mixmlpem( net, x, t, options, n,'scg');
save ([out,'_trained_net'],'net');
fprintf('\n Mixture of Experts model file name: ');
ME_model = [out,'_trained_net.mat'];
disp(ME_model);

% [v, m] = TrainMixtureOfExperts('regression',moetype,x,t,nexp,max_iter,0.1,0.98);
%%%%%%%%% LOGISTIC REGRESSION Train%%%%%%%%%%%%
[b_train_ori,~,stats] = glmfit(x,t,'binomial','link','logit');
b_train = b_train_ori(2:end);
pval = stats.p;
p_value = stats.p(2:end);
ind_pv = p_value<=p_val;
b_train_pv = b_train(ind_pv);
x_train_pv = x(:,ind_pv);
ax3_train = bsxfun(@times,b_train_pv',x_train_pv);
s2_train = sum(ax3_train,2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y_train = mixmlpfwd(net,x);
save([out,'_train_score.txt'], 'y_train', '-ascii');
[x_ME_train,y_ME_train,~,AUC_ME_train] = perfcurve(t,y_train',size(t,2));
[x_logistic_train,y_logistic_train,~,AUC_Logistic_train] = perfcurve(t,s2_train',size(t,2));
if strcmp(log_reg,'on')
    fprintf('\n AUC of the Train Model for ME = %f\r', AUC_ME_train);
    fprintf('\n AUC of the Train Model for Logistic regression = %f\r', AUC_Logistic_train);
    fprintf('\n');
    plot(x_ME_train, y_ME_train,'m-', x_logistic_train, y_logistic_train, 'b-');
    hold off;
    xlabel_train = xlabel('1-Specificity');
    ylabel_train = ylabel('Sensitivity');
    set(xlabel_train,'FontSize',16);
    set(ylabel_train,'FontSize',16);
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    grid on;
    box on;
    title_t = title(sprintf(title_me));
    set(title_t,'FontSize',16);
    legend_Train = legend(sprintf('AUC-ME = %4.3f',AUC_ME_train), sprintf('AUC-LOG = %4.3f',AUC_Logistic_train),'Location','SouthEast');
    set(legend_Train,'FontSize',16);
    fprintf(' logistic regression model file name: ');
    logistic_regression_model = [out,'_logistic_regression_model.txt'];
    disp(logistic_regression_model);
    save([out,'_logistic_regression_model.txt'],'b_train_ori','-ascii');
    fprintf('\n logistic regression model p_value file name: ');
    logistic_regression_pvalue = [out,'_logistic_p_value.txt'];
    disp(logistic_regression_pvalue);
    save([out,'_logistic_p_value.txt'],'pval','-ascii');
else
    fprintf('\n AUC of the Train Model for ME = %f\r', AUC_ME_train);
    fprintf('\n AUC of the Train Model for Logistic regression = %f\r', AUC_Logistic_train);
    plot(x_ME_train, y_ME_train, 'm-');
    xlabel_train = xlabel('1-Specificity');
    ylabel_train = ylabel('Sensitivity');
    set(xlabel_train,'FontSize',16);
    set(ylabel_train,'FontSize',16);
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    grid on;
    box on;
    title_t = title(sprintf(title_me));
    set(title_t,'FontSize',16);
    legend_svm_Train = legend(sprintf('AUC-ME = %4.3f',AUC_ME_train),'Location','SouthEast');
    set(legend_svm_Train,'FontSize',16);
end
output_train = [out, '_train_ROC'];
export_fig(output_train, '-pdf', '-eps');
end