function SVM_train_test2(varargin)
fprintf('\n Example command line: "./presult train_test2 SVM --train_data <train_data.txt> --test_data <test_data.txt>" \n')
p = inputParser;
addParameter(p,'train_data','pid_train_data.txt',@ischar);
addParameter(p,'test_data','pid_test_data.txt',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'var','variable.txt',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'out','SVM',@ischar);
addParameter(p,'kernel_function','linear',@ischar); % 'rbf' or 'linear'
addParameter(p,'iters_limit','300'); % optimize iterations limit for input data
addParameter(p,'box_constraint','1'); % BoxConstraint 1 for classification 
addParameter(p,'CacheSize','maximum',@ischar);
addParameter(p,'randm','shuffle',@ischar);
addParameter(p,'logistic_regression','on',@ischar);
addParameter(p,'pvalue_logistic','0.01');
addParameter(p,'title','Support Vector Mchine',@ischar);

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
box_constraint = (p.Results.box_constraint);
kernel_function = (p.Results.kernel_function);
iters_limit = (p.Results.iters_limit);
miss = (p.Results.missing_value);
out = (p.Results.out);
ran = (p.Results.randm);
log_reg = (p.Results.logistic_regression);
p_val_log = (p.Results.pvalue_logistic);
title_svm = (p.Results.title);

box_constraint = str2double(box_constraint);
iters_limit = str2double(iters_limit);
p_val = str2double(p_val_log);
if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', train_file, '--output', 'normalized_train_data_tt2_SVM.txt','--var',var);
else
   perl('newest_normalization.pl', '--input', train_file, '--output', 'normalized_train_data_tt2_SVM.txt', '--csv','--var',var);
end


% q = load('normalized_train_data_svm.txt');


if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', test_file, '--output', 'normalized_test_data_tt2_SVM.txt','--var',var);
else
   perl('newest_normalization.pl', '--input', test_file, '--output', 'normalized_test_data_tt2_SVM.txt', '--csv','--var',var);
end
q = load('normalized_train_data_tt2_SVM.txt');

%%%%%%%%%%%%%%%%%%%%
%DataSorting start
%%%%%%%%%%%%%%%%%%%%
rownumbers = size(q,1);
rng(ran);
ran = rand(rownumbers,1);
data_file2 = [ran q];
data_file3 = sortrows(data_file2,1);
data_file = data_file3(:,2:end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ninput = size(data_file,2);
x =data_file(:,1:ninput-1);
t =data_file(:,ninput);
SVMModel = fitcsvm(x,t,'KernelFunction',kernel_function,'Standardize',true, 'IterationLimit',iters_limit,'BoxConstraint',box_constraint,'Verbose',0);
test = load('normalized_test_data_tt2_SVM.txt');
ninput = size(test,2);
xtest = test(:,1:ninput-1);
ttest = test(:,ninput);
ScoreSVMModel = fitSVMPosterior(SVMModel,x,t);
save ([out,'_trained_net_tt2'],'ScoreSVMModel');
[~,score_test] = predict(ScoreSVMModel,xtest);
ytest = score_test(:,2);
[~,score_train] = predict(ScoreSVMModel,x);
ytrain = score_train(:,2);
save([out,'_test_score_tt2.txt'], 'ytest', '-ascii');
save([out,'_train_score_tt2.txt'], 'ytrain', '-ascii');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x_svm_train,y_svm_train,~,AUC_svm_train] = perfcurve(t,ytrain',size(t,2));
[x_svm_test,y_svm_test,~,AUC_svm_test] = perfcurve(ttest,ytest',size(ttest,2));
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
fprintf('\n AUC of the Train Model = %4.3f\r', AUC_svm_train);
fprintf('\n AUC of the Test Model = %4.3f\r', AUC_svm_test);

% %%%%%%%%% LOGISTIC REGRESSION TRAIN%%%%%%%%%%%%

[b_train_ori,~,stats] = glmfit(x,t,'binomial','link','logit');
b_train = b_train_ori(2:end);
p_value = stats.p;
stats.p = stats.p(2:end);
ind_pv = stats.p<=p_val;
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
[x_logistic_test,y_logistic_test,~,AUC_Logistic_test] = perfcurve(ttest,s2_test',size(ttest,2));

if strcmp(log_reg,'on');
   plot(x_svm_train,y_svm_train,x_svm_test,y_svm_test,x_logistic_train,y_logistic_train,x_logistic_test,y_logistic_test,'-');
   xlabel_tt2 = xlabel('1-Specificity');
   set(xlabel_tt2,'FontSize',16);
   ylabel_tt2 = ylabel('Sensitivity');
   set(ylabel_tt2,'FontSize',16);
   set(gcf,'color','w');
   set(gca,'FontSize',16);
   grid on;
   fprintf('\n AUC of the Logistic Train Model = %4.3f\r', AUC_Logistic_train);
   fprintf('\n AUC of the Logistic Test Model = %4.3f\r', AUC_Logistic_test);
   title_tt2 = title(sprintf(title_svm));
   set(title_tt2,'FontSize',16);
   legend_tt2 = legend(sprintf('train-SVM =%4.3f',AUC_svm_train), sprintf('test-SVM =%4.3f',AUC_svm_test), sprintf('train-log =%4.3f',AUC_Logistic_train), sprintf('test-log =%4.3f',AUC_Logistic_test),'Location','SouthEast');
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
   plot(x_svm_train,y_svm_train,x_svm_test,y_svm_test,'-');
   xlabel_tt2 = xlabel('1-Specificity');
   set(xlabel_tt2,'FontSize',16);
   ylabel_tt2 = ylabel('Sensitivity');
   set(ylabel_tt2,'FontSize',16);
   set(gcf,'color','w');
   set(gca,'FontSize',16);
   grid on;
   title_tt2 = title(sprintf(title_svm));
   set(title_tt2,'FontSize',16);
   legend_tt2 = legend(sprintf('train-SVM =%4.3f',AUC_svm_train), sprintf('test-SVM =%4.3f',AUC_svm_test),'Location','SouthEast');
   set(legend_tt2,'FontSize',16);
end
graph = [out,'_train_test2'];
export_fig(graph, '-pdf', '-eps');
end