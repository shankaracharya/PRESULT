function SVM_train(varargin)
fprintf('\n Example command line: "./presult train SVM --train_data <training_data.txt>" \n');
p = inputParser;
%disp(varargin(:));
addParameter(p,'train_data','pid_train_data.txt',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'var','variable.txt',@ischar)
addParameter(p,'box_constraint','1'); % BoxConstraint 1 for classification 
addParameter(p,'CacheSize','maximum',@ischar);
addParameter(p,'kernel_function','linear',@ischar); % 'rbf' or 'linear'
addParameter(p,'iters_limit','300'); % optimize iterations limit for input data
addParameter(p,'missing_value','?');
addParameter(p,'out','SVM',@ischar);
addParameter(p,'randm','shuffle',@ischar);
addParameter(p,'logistic_regression','on',@ischar);
addParameter(p,'title','Support Vector Machine',@ischar);

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
csv = (p.Results.csv);
var = (p.Results.var);
miss = (p.Results.missing_value);
log_reg = (p.Results.logistic_regression);
box_constraint = (p.Results.box_constraint);
kernel_function = (p.Results.kernel_function);
iters_limit = (p.Results.iters_limit);
title_svm = (p.Results.title);
box_constraint = str2double(box_constraint);
iters_limit = str2double(iters_limit);
out = (p.Results.out);
ran = (p.Results.randm);
if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', train_file, '--output', 'SVM_normalized_train_data.txt','--var',var);
else
   perl('newest_normalization.pl', '--input', train_file, '--output', 'SVM_normalized_train_data.txt', '--csv','--var',var);
end
q = load('SVM_normalized_train_data.txt');

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
%Data Sorting end_
%%%%%%%%%%%%%%%%%%%%%%
ninput = size(data_file,2);
x = data_file(:,1:ninput-1);
t = data_file(:,ninput);
SVMModel = fitcsvm(x,t,'KernelFunction',kernel_function,'Standardize',true, 'IterationLimit',iters_limit,'BoxConstraint',box_constraint,'Verbose',0);
ScoreSVMModel = fitSVMPosterior(SVMModel,x,t);
save ([out,'_trained_net'],'ScoreSVMModel');
[~,score_train] = predict(ScoreSVMModel,x);
y_SVM_train = score_train(:,2);
save([out,'_train_score.txt'], 'y_SVM_train', '-ascii');
%%%%%%%%% LOGISTIC REGRESSION %%%%%%%%%%%%
[b_train_ori,~,stats] = glmfit(x,t,'binomial','link','logit');
b_train = b_train_ori(2:end);
pval = stats.p;
p_value = stats.p(2:end);
ind_pv = p_value<=0.01;
b_train_pv = b_train(ind_pv);
x_train_pv = x(:,ind_pv);
ax3_train = bsxfun(@times,b_train_pv',x_train_pv);
s2_train = sum(ax3_train,2);

%%%%%%%% Logistic regression %%%%%%%%%%

[x_train,y_train,~,AUC_SVM_train] = perfcurve(t, y_SVM_train',size(t,2));
[x_logistic_train,y_logistic_train,~,AUC_Logistic_train] = perfcurve(t,s2_train',size(t,2));

if strcmp(log_reg,'on');
    fprintf('\n AUC of the Train Model for SVM = %4.3f\r', AUC_SVM_train);
    fprintf('\n AUC of the Train Model for Logistic regression = %4.3f\r', AUC_Logistic_train);
    plot(x_train,y_train,'m-',x_logistic_train,y_logistic_train, 'b-');
    hold off;
    box on;
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    xlabel_train = xlabel('1-Specificity');
    ylabel_train = ylabel('Sensitivity');
    set(xlabel_train,'FontSize',16);
    set(ylabel_train,'FontSize',16);
    grid on;
    title_t = title(sprintf(title_svm));
    set(title_t,'FontSize',16);
    legend_svm_Train = legend(sprintf('AUC-SVM = %4.3f',AUC_SVM_train), sprintf('AUC-LOG = %4.3f',AUC_Logistic_train),'Location','SouthEast');
    set(legend_svm_Train,'FontSize',16);
    fprintf('\n logistic regression model file name: ');
    logistic_regression_model = [out,'_logistic_regression_model.txt'];
    disp(logistic_regression_model);
    save([out,'_logistic_regression_model.txt'],'b_train_ori','-ascii');
    fprintf('\n logistic regression model p_value file name: ');
    logistic_regression_pvalue = [out,'_logistic_p_value.txt'];
    disp(logistic_regression_pvalue);
    save([out,'_logistic_p_value.txt'],'p_value','-ascii');
else
    fprintf('\n AUC of the Train Model for SVM = %4.3f\r', AUC_SVM_train);
    fprintf('\n AUC of the Train Model for Logistic regression = %4.3f\r', AUC_Logistic_train);
    strcmp(log_reg,'off');
    plot(x_train,y_train, 'b-');
    xlabel_train = xlabel('1-Specificity');
    ylabel_train = ylabel('Sensitivity');
    set(xlabel_train,'FontSize',16);
    set(ylabel_train,'FontSize',16);
    box on;
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    grid on;
    title_t = title(sprintf(title_svm));
    set(title_t,'FontSize',16);
    legend_svm_Train = legend(sprintf('AUC-SVM = %4.3f',AUC_SVM_train),'Location','SouthEast');
    set(legend_svm_Train,'FontSize',16);
end
output_train = [out, '_train_ROC'];
export_fig (output_train, '-pdf', '-eps');
end