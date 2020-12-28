function RT_test(varargin)
fprintf('\n Example command line: "./presult test RT --test_data <test_data.txt> --trained_net <RT_trained_net.mat> --logistic_regression <on/off> --log_train_model <logistic_regression_model.txt> --log_p_value <logistic_p_value.txt>" \n');
p = inputParser;
addParameter(p,'test_data','dia_test_data.txt',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'var','variable.txt',@ischar)
addParameter(p,'trained_net','RT_trained_net.mat',@ischar);
addParameter(p,'log_train_model','RT_logistic_regression_model.txt',@ischar);
addParameter(p,'log_p_value','RT_logistic_p_value.txt',@ischar);
addParameter(p,'out','RT',@ischar);
addParameter(p,'logistic_regression','off',@ischar);
addParameter(p,'title','Regression Tree',@ischar);

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
var = (p.Results.var);
miss = (p.Results.missing_value);
rtree = (p.Results.trained_net);
log_out = (p.Results.log_train_model);
log_p_value = (p.Results.log_p_value);
out = (p.Results.out);
log_reg = (p.Results.logistic_regression);
title_rt = (p.Results.title);
if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', test_file, '--output', 'normalized_test_data_Test_RT.txt','--var',var);
else
   perl('newest_normalization.pl', '--input', test_file, '--output', 'normalized_test_data_Test_RT.txt', '--csv','--var',var);
end
test = load('normalized_test_data_Test_RT.txt');
ninput = size(test,2);
xtest = test(:,1:ninput-1);
ttest = test(:,ninput);
load(rtree);

y_RT_test = predict(rtree,xtest);
save([out,'_test_score.txt'], 'y_RT_test', '-ascii');
[C]=confmat(y_RT_test,ttest);
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

[x_test,y_test,~,AUC_RT_test] = perfcurve(ttest,y_RT_test',size(ttest,2));


if strcmp(log_reg,'on');
    train_out = load(log_out,'-ascii');
    logistic_p_value = load(log_p_value,'-ascii');
    %%%%%%%%% LOGISTIC REGRESSION TEST%%%%%%%%%%%%
    ind_pv = logistic_p_value<=0.01;
    xxtest = xtest(:,ind_pv);
    train_out = train_out(ind_pv);
    ax3_test = bsxfun(@times,train_out',xxtest);
    s2_test = sum(ax3_test,2);
    [x_logistic_test,y_logistic_test,~,AUC_Logistic_test] = perfcurve(ttest,s2_test',size(ttest,2));
    %%%%%%%% Logistic regression %%%%%%%%%%
    plot(x_test,y_test,'m-',x_logistic_test,y_logistic_test, 'b-');
    legend_RT_test = legend(sprintf('AUC-RT = %4.2f',AUC_RT_test), sprintf('AUC-LOG = %4.2f', AUC_Logistic_test),'Location','SouthEast');
    set(legend_RT_test,'FontSize',16)
    xlabel_RT_test = xlabel('1-Specificity');
    set(xlabel_RT_test,'FontSize',16);
    ylabel_RT_test = ylabel('Sensitivity');
    set(ylabel_RT_test,'FontSize',16);
    grid on;
    box on;
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    title_te = title(sprintf(title_rt));
    set(title_te,'FontSize',16);
    fprintf('\n AUC of the Test Model for RT = %4.3f\r\n', AUC_RT_test);
    fprintf('\n AUC of the Test Model for Logistic regression = %4.3f\r\n', AUC_Logistic_test);
else
    plot(x_test,y_test, 'm-');
    legend(['AUC-RT = ',num2str(AUC_RT_test)],'Location','SouthEast');
    xlabel_RT_test = xlabel('1-Specificity');
    set(xlabel_RT_test,'FontSize',16);
    ylabel_RT_test = ylabel('Sensitivity');
    set(ylabel_RT_test,'FontSize',16);
    grid on;
    box on;
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    title_te = title(sprintf(title_rt));
    set(title_te,'FontSize',16);
    fprintf('\n AUC of the Test Model for RT = %4.3f\r\n', AUC_RT_test);
end
output_test = [out, '_test_ROC'];
export_fig(output_test, '-pdf', '-eps');
end