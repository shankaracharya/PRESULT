function ME_test(varargin)
fprintf('\n Example command line: "./presult test ME --test_data <test_data.txt> --trained_net <ME_trained_net.mat> --logistic_regression <on/off> --log_train_model <logistic_regression_model> --log_p_value <logistic_regression_p_value>" \n');
p = inputParser;
addParameter(p,'test_data','test_data',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'trained_net','ME_trained_net.mat',@ischar);
addParameter(p,'var','variables.txt',@ischar);
addParameter(p,'log_train_model','ME_logistic_regression_model.txt',@ischar);
addParameter(p,'log_p_value','ME_logistic_p_value.txt',@ischar);
addParameter(p,'out','out',@ischar);
addParameter(p,'logistic_regression','off',@ischar);
addParameter(p,'title','Mixture of Experts',@ischar);
addParameter(p,'p_val','0.01');

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

parse(p,new_array{:});
test_file = (p.Results.test_data);
csv = (p.Results.csv);
miss = (p.Results.missing_value);
var = (p.Results.var);
net = (p.Results.trained_net);
log_out = (p.Results.log_train_model);
log_p_value = (p.Results.log_p_value);
out = (p.Results.out);
log_reg = (p.Results.logistic_regression);
title_me = (p.Results.title);
p_val = (p.Results.p_val);
pval = str2double(p_val);


if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', test_file, '--output', 'normalized_test_data_Test_ME.txt','--var',var);

else
   perl('newest_normalization.pl', '--input', test_file, '--output', 'normalized_test_data_Test_ME.txt', '--csv','--var',var);
end

test = load('normalized_test_data_Test_ME.txt');
ninput = size(test,2);
xtest = test(:,1:ninput-1);
ttest = test(:,ninput);
load(net,'-mat');

y_test = mixmlpfwd(net,xtest);
save([out,'_test_score.txt'], 'y_test', '-ascii');
[C]=confmat(y_test,ttest);
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

fprintf('\n Sensitivity of the Test Model = %4.3f\r\n', Sensitivity);
fprintf('\n Specificity of the Test Model = %4.3f\r\n', Specificity);
fprintf('\n Accuracy of the Test Model = %4.3f\r\n', Accuracy);


[x_ME_test,y_ME_test,~,AUC_ME_test] = perfcurve(ttest,y_test',size(ttest,2));



if strcmp(log_reg,'on')
    train_out = load(log_out,'-ascii');
    %%%%%%%%% LOGISTIC REGRESSION TEST%%%%%%%%%%%%
    logistic_p_value = load(log_p_value,'-ascii');
    ind_pv = logistic_p_value<=pval;
    xxtest = xtest(:,ind_pv);
    train_out = train_out(ind_pv);
    ax3_test = bsxfun(@times,train_out',xxtest);
    s2_test = sum(ax3_test,2);
    [x_logistic_test,y_logistic_test,~,AUC_Logistic_test] = perfcurve(ttest,s2_test',size(ttest,2));
    %%%%%%%% Logistic regression %%%%%%%%%%
    plot(x_ME_test,y_ME_test,'m-',x_logistic_test,y_logistic_test, 'b-');
    legend_test = legend(sprintf('AUC-ME =%4.3f',AUC_ME_test), sprintf('AUC-LOG =%4.3f',AUC_Logistic_test),'Location','SouthEast');
    set(legend_test,'FontSize',14);
    xlabel_train = xlabel('1-Specificity');
    ylabel_train = ylabel('Sensitivity');
    set(xlabel_train,'FontSize',16);
    set(ylabel_train,'FontSize',16);
    grid on;
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    title_te = title(sprintf(title_me));
    set(title_te,'FontSize',16);
    fprintf('\n AUC of the Test Model for ME = %4.3f\r\n', AUC_ME_test);
    fprintf('\n AUC of the Test Model for Logistic regression = %4.3f\r\n', AUC_Logistic_test);
else
    fprintf('\n AUC of the Test Model for ME = %4.3f\r\n', AUC_ME_test);
    plot(x_ME_test,y_ME_test, 'm-');
    legend_test = legend(sprintf('AUC-ME =%4.3f',AUC_ME_test),'Location','SouthEast');
    set(legend_test,'FontSize',16);
    xlabel_train = xlabel('1-Specificity');
    ylabel_train = ylabel('Sensitivity');
    set(xlabel_train,'FontSize',16);
    set(ylabel_train,'FontSize',16);
    grid on;
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    title_te = title(sprintf(title_me));
    set(title_te,'FontSize',16);
end
output_test = [out, '_test_ROC'];
export_fig(output_test, '-pdf', '-eps');
end