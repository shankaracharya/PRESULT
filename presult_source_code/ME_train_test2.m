function ME_train_test2(varargin)
fprintf('\n Example command line: "./presult train_test2 ME --train_data <train_data.txt> --test_data <test_data.txt>" \n')
p = inputParser;
addParameter(p,'train_data','pid_train_data.txt',@ischar);
addParameter(p,'test_data','pid_test_data.txt',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'var','variables.txt',@ischar);
addParameter(p,'emiters','6');
addParameter(p,'actfunc','logistic',@ischar);
addParameter(p,'out','ME',@ischar);
addParameter(p,'hexp','8');
addParameter(p,'hgate','8');
addParameter(p,'mstep_iters','8');
addParameter(p,'nexp','5');
addParameter(p,'randm','shuffle',@ischar);
addParameter(p,'logistic_regression','on',@ischar);
addParameter(p,'pvalue_logistic','0.05');
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
test_file = (p.Results.test_data);
csv = (p.Results.csv);
var = (p.Results.var);
miss = (p.Results.missing_value);
n = (p.Results.emiters);
actfunc = (p.Results.actfunc);
hexp = (p.Results.hexp);
out = (p.Results.out);
hgate = (p.Results.hgate);
nexp = (p.Results.nexp);
ran = (p.Results.randm);
mstep_iters = (p.Results.mstep_iters);
log_reg = (p.Results.logistic_regression);
p_val = (p.Results.pvalue_logistic);
title_me = (p.Results.title);

n = str2double(n);
hexp = str2double(hexp);
hgate = str2double(hgate);
nexp = str2double(nexp);
mstep_iters = str2double(mstep_iters);
p_val_log = str2double(p_val);

if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', train_file, '--output', 'ME_tt2_normalized_train_data.txt','--var',var);
else
   perl('newest_normalization.pl', '--input', train_file, '--output', 'ME_tt2_normalized_train_data.txt', '--csv','--var',var);
end


if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', test_file, '--output', 'ME_tt2_normalized_test_data.txt');
else
   perl('newest_normalization.pl', '--input', test_file, '--output', 'ME_tt2_normalized_test_data.txt', '--csv');
end

q = load('ME_tt2_normalized_train_data.txt');

% % % % %%%%%%%%Spearman%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%
%DataSorting start
%%%%%%%%%%%%%%%%%%%%
rownumbers = size(q,1);
rng(ran);
ran = rand(rownumbers,1);
data_file2 = [ran q];
data_file3 = sortrows(data_file2,1);
data_file = data_file3(:,2:end);
%%%%%%%%%%%%%%%%%%%%%
ninput = size(data_file,2);
x = data_file(:,1:ninput-1);
t = data_file(:,ninput);
net =  mixmlp(ninput-1,hexp,1,hgate,nexp,actfunc,'standard');
options  = zeros(1,18);
options(1)  =  -1;  % no display of the error value within the M-step 
options(14) =  mstep_iters;  % the number of iterations in the M-step   
[net,~] = mixmlpem( net, x, t, options, n,'scg');
save ([out,'_trained_net_tt2'],'net');
fprintf('\n Mixture of Experts model file name: ');
ME_model = [out,'_trained_net_tt2.mat'];
disp(ME_model);
test = load('ME_tt2_normalized_test_data.txt');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ninput = size(test,2);
xtest = test(:,1:ninput-1);
ttest = test(:,ninput);
ytest = mixmlpfwd(net,xtest);
ytrain = mixmlpfwd(net,x);
save([out,'_test_score_tt2.txt'], 'ytest', '-ascii');
save([out,'_train_score_tt2.txt'], 'ytrain', '-ascii');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x_ME_train, y_ME_train,~,AUC_ME_train] = perfcurve(t,ytrain',size(t,2));
[x_ME_test, y_ME_test,~,AUC_ME_test] = perfcurve(ttest,ytest',size(ttest,2));
grid on;

[C]=confmat(ytest,ttest);
fprintf('\n Confusion Matrix of the Test Model: \r\n');
disp(C);
C1 = C(1,1);
C2 = C(1,2);
C3 = C(2,1);
C4 = C(2,2);
Correct_prediction = C1+C4;
Wrong_prediction = C2+C3;
Accuracy = (Correct_prediction/(Correct_prediction + Wrong_prediction))*100;
Sensitivity = C1/(C1+C3);
Specificity = C4/(C4+C2);
fprintf(' Sensitivity of the Test Model = %4.3f\r', Sensitivity);
fprintf('\n Specificity of the Test Model = %4.3f\r', Specificity);
fprintf('\n Accuracy of the Test Model = %4.3f\r', Accuracy);
fprintf('\n AUC of the Train Model = %4.3f\r', AUC_ME_train);
fprintf('\n AUC of the Test Model = %4.3f\r', AUC_ME_test);


% %%%%%%%%% LOGISTIC REGRESSION TRAIN%%%%%%%%%%%%

[b_train_ori,~,stats] = glmfit(x,t,'binomial','link','logit');
b_train = b_train_ori(2:end);
p_value = stats.p;
stats.p = stats.p(2:end);
ind_pv = stats.p<=p_val_log;
x_train = x(:,ind_pv);
b_train_pv = b_train(ind_pv);
ax3_train = bsxfun(@times,b_train_pv',x_train);
s2_train = sum(ax3_train,2);

%%%%%%%%% LOGISTIC REGRESSION TEST%%%%%%%%%%%%
[x_logistic_train,y_logistic_train,~,AUC_Logistic_train] = perfcurve(t,s2_train',size(t,2));
xxtest = xtest(:,ind_pv);
bb_train = b_train(ind_pv);
ax3_test = bsxfun(@times,bb_train',xxtest);
s2_test = sum(ax3_test,2);


if strcmp(log_reg,'on');
   [x_logistic_test,y_logistic_test,~,AUC_Logistic_test] = perfcurve(ttest,s2_test',size(ttest,2));
   fprintf('\n AUC of the Logistic Train Model = %4.3f\r', AUC_Logistic_train);
   fprintf('\n AUC of the Logistic Test Model = %4.3f\r', AUC_Logistic_test);
   plot(x_ME_train,y_ME_train,x_ME_test,y_ME_test,x_logistic_train,y_logistic_train,x_logistic_test,y_logistic_test,'-');
   hold off;
   xlabel_tt2 = xlabel('1-Specificity');
   set(xlabel_tt2,'FontSize',16);
   ylabel_tt2 = ylabel('Sensitivity');
   set(ylabel_tt2,'FontSize',16);
   set(gcf,'color','w');
   set(gca,'FontSize',16);
   grid on;
   title_tt2 = title(sprintf(title_me));
   set(title_tt2,'FontSize',16);
   legend_tt2 = legend(sprintf('train-ME = %4.3f',AUC_ME_train), sprintf('test-ME = %4.3f',AUC_ME_test), sprintf('train-log = %4.3f',AUC_Logistic_train), sprintf('test-log = %4.3f',AUC_Logistic_test),'Location','SouthEast');
   set(legend_tt2,'FontSize',16);
   save([out,'_logistic_regression_model.txt'],'b_train_ori','-ascii');
   save([out,'_logistic_p_value.txt'],'p_value','-ascii');
   fprintf('\n logistic regression model file name: ');
   logistic_regression_model = [out,'_logistic_regression_model.txt'];
   disp(logistic_regression_model);
   fprintf('\n logistic regression model p_value file name: ');
   logistic_regression_pvalue = [out,'_logistic_p_value.txt'];
   disp(logistic_regression_pvalue);
else
   strcmp(log_reg,'off');
   plot(x_ME_train,y_ME_train,x_ME_test,y_ME_test,'-');
   xlabel_tt2 = xlabel('1-Specificity');
   set(xlabel_tt2,'FontSize',14);
   ylabel_tt2 = ylabel('Sensitivity');
   set(ylabel_tt2,'FontSize',14);
   set(gcf,'color','w');
   set(gca,'FontSize',16);
   grid on;
   title_tt2 = title(sprintf(title_me));
   set(title_tt2,'FontSize',16);
   legend_tt2 = legend(['train-ME = ',num2str(AUC_ME_train)], ['test-ME = ',num2str(AUC_ME_test)],'Location','SouthEast');
   set(legend_tt2,'FontSize',16);
end
graph = [out,'_train_test2'];
export_fig(graph, '-pdf', '-eps');
end