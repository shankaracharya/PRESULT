function Logistic_kfold(varargin)
p = inputParser;
addParameter(p,'input_data','pid_raw_data.txt',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'kfold','10');
addParameter(p,'out','LR',@ischar);
addParameter(p,'randm','shuffle',@ischar);
addParameter(p,'title','Logistic Regression',@ischar);

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
data_file = (p.Results.input_data);
kvalue = (p.Results.kfold);
csv = (p.Results.csv);
miss = (p.Results.missing_value);
out = (p.Results.out);
ran = (p.Results.randm);
title_lr = (p.Results.title);
if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', data_file, '--output', 'normalized_data_Logistic.txt');
else
   perl('newest_normalization.pl', '--input', data_file, '--output', 'normalized_data_Logistic.txt', '--csv');
end
data_file1 = load('normalized_data_Logistic.txt');
k = str2double(kvalue);
rownumbers = size(data_file1,1);
%DataSorting start
%%%%%%%%%%%%%%%%%%%%
rng(ran);
ran = rand(rownumbers,1);
data_file2 = [ran data_file1];
data_file3 = sortrows(data_file2,1);
data_file = data_file3(:,2:end);
%%%%%%%%%%%%%%%%%%%%%
%Data Sorting end
%%%%%%%%%%%%%%%%%%%%%%
chunk = floor(rownumbers/k);
hold all;
grid on;
ah1 = gca;
% disp(mean(data_file));
for f = 1:k
        test = data_file((f-1)*chunk+1:(f)*chunk,:);
        train = [data_file(1:(f-1)*chunk,:); data_file(f*chunk+1:end, :)];
        disp(mean(test));
        disp(mean(train));
        save ([out,'_train_data_',int2str(f)],'train','-ascii');
        save ([out,'_test_data_',int2str(f)],'test','-ascii');
        
        train_load = load([out,'_train_data_',int2str(f)]);
        ninput = size(train_load,2);
        x = train_load(:,1:ninput-1);
        t = train_load(:,ninput);
        [b_train,dev,stats] = glmfit(x,t,'binomial','link','logit');
        b_train = b_train(2:end);
        p_val = stats.p(2:end);
        ind_p = p_val<=0.01;
        bb_train = b_train(ind_p);
        test_load = load([out,'_test_data_',int2str(f)]);
        ninput = size(test_load,2);
        xtest = test_load(:,1:ninput-1);
        ttest = test_load(:,ninput);
        xxtest = xtest(:,ind_p);
        ax3_test = bsxfun(@times,bb_train',xxtest);
        s2_test = sum(ax3_test,2);
        [x_logistic_test,y_logistic_test,T_logistic_test,AUC_Logistic_test] = perfcurve(ttest,s2_test',size(ttest,2));
        plot(x_logistic_test,y_logistic_test,'-');
        legendinfo{f} = ['AUC = ' num2str(AUC_Logistic_test)];
        AUC_COL_TEST(f,:) = AUC_Logistic_test;
        
%%%%%%%%%%%%%%% sensitivity, specificity and accuracy %%%%%%%%%%%%
        [C]=confmat(s2_test,ttest);
        fprintf('\n Confusion Matrix of the Test Model: \r\n');
        disp(C);
        C1 = C(1,1);
        C2 = C(1,2);
        C3 = C(2,1);
        C4 = C(2,2);
        Correct_prediction = C1+C4;
        Wrong_prediction = C2+C3;
        Sensitivity = C1/(C1+C3);
        Specificity = C4/(C4+C2);
        Accuracy = (Correct_prediction/(Correct_prediction + Wrong_prediction))*100;
        fprintf('\n Sensitivity of the Test Model = %f\r\n', Sensitivity);
        fprintf('\n Specificity of the Test Model = %f\r\n', Specificity);
        fprintf('\n Accuracy of the Test Model = %f\r\n', Accuracy);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
ACCU(f,:) = Accuracy;
SENS(f,:) = Sensitivity;
SPEC(f,:) = Specificity;
kfold(f,:) = f;
% fprintf('\n ############ completion of %d fold cross validation ############### \n',f);

end
hold off;
Avg_AUC_test = mean(AUC_COL_TEST);
legend(ah1, legendinfo,'Location','SouthEast','FontSize',14);
xlabel_kf = xlabel(ah1,'1-Specificity');
set(xlabel_kf,'FontSize',16);
ylabel_kf = ylabel(ah1, 'Sensitivity');
set(ylabel_kf,'FontSize',16);
set(gcf,'color','w');
set(gca,'FontSize',16);
box on;
dim =[0.63 0.4 0.1 0.2];
str = ['Avg AUC = ',num2str(Avg_AUC_test)];
annotation('textbox',dim,'String',str,'FitBoxToText','on','FontSize',14,'Color','k');
% Avg_AUC_train = mean(AUC_COL_TRAIN);
Average_sesitivity = mean(SENS);
Average_specificity = mean(SPEC);
Average_accuracy = mean(ACCU);
Avg_AUC_test = mean(AUC_COL_TEST);
title_kf = title(ah1, sprintf(title_lr));
set(title_kf,'FontSize',16);
fprintf('\n ############ Summary for %d-fold cross validation ############### \n',k);
summary = [kfold SENS SPEC ACCU AUC_COL_TEST];
%     disp(KFOLD);
TT = array2table(summary,'VariableNames',{'kfold' 'Sensitivity' 'Specificity' 'Accuracy' 'AUC'});
disp(TT);
fprintf('\n Average AUC of the Test Model = %f\r\n', Avg_AUC_test);
fprintf('\n Average Sensitivity of the Test Model = %f\r\n', Average_sesitivity);
fprintf('\n Average Specificity of the Test Model = %f\r\n', Average_specificity);
fprintf('\n Average Accuracy of the Test Model = %f\r\n', Average_accuracy);
fprintf('\n Average AUC of the Test Model = %f\r\n', Avg_AUC_test);
kfold_out = [out,'_kfold_ROC'];
export_fig (kfold_out, '-pdf', '-eps');
end
        
        
