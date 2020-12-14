function SVM_kfold(varargin)
fprintf('\n Example command line: "./presult kfold SVM --input_data <input_data.txt> --randm <yes/no> --kfold <int>" \n')
p = inputParser;
addParameter(p,'input_data','pid_raw_data.txt',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'var','variable.txt',@ischar)
addParameter(p,'missing_value','?');
addParameter(p,'kfold','10');
addParameter(p,'box_constraint','1'); % BoxConstraint 1 for classification 
addParameter(p,'CacheSize','maximum',@ischar);
addParameter(p,'kernel_function','linear',@ischar); % 'rbf' or 'linear'
addParameter(p,'out','SVM',@ischar);
addParameter(p,'iters_limit','2000'); % optimize iterations limit for input data
addParameter(p,'randm','shuffle',@ischar);
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
data_file = (p.Results.input_data);
csv = (p.Results.csv);
var = (p.Results.var);
miss = (p.Results.missing_value);
kvalue = (p.Results.kfold);
box_constraint = (p.Results.box_constraint);
kernel_function = (p.Results.kernel_function);
iters_limit = (p.Results.iters_limit);
out = (p.Results.out);
ran = (p.Results.randm);
title_svm = (p.Results.title);

if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', data_file, '--output', 'normalized_data_kfold_svm.txt','--var',var);
else
   perl('newest_normalization.pl', '--input', data_file, '--output', 'normalized_data_kfold_svm.txt', '--csv','--var',var);
end
data_file1 = load('normalized_data_kfold_svm.txt');
k = str2double(kvalue);
box_constraint = str2double(box_constraint);
iters_limit = str2double(iters_limit);
rownumbers = size(data_file1,1);
%%%%%%%%%%%%%%%%%%%%
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
    for f = 1:k
        fprintf('\n ############ %d-fold cross validation ############### \n',f);
        test = data_file((f-1)*chunk+1:(f)*chunk,:);
        train = [data_file(1:(f-1)*chunk,:); data_file(f*chunk+1:end, :)];
        save ([out,'_train_data_kfold_',int2str(f),'.txt'],'train','-ascii');
        save ([out,'_test_data_kfold_',int2str(f),'.txt'],'test','-ascii');
        fprintf(['\n Train data for ', int2str(f), '-kfold ', 'is saved as: ']);
        fprintf([out,'_train_data_kfold_',int2str(f),'.txt'],'\n');
        fprintf('\n');
        fprintf(['\n Test data for ', int2str(f), '-kfold ', 'is saved as: ']);
        fprintf([out,'_test_data_kfold_',int2str(f),'.txt'],'\n');
        fprintf('\n');
        q = load([out,'_train_data_kfold_',int2str(f),'.txt']);
        ninput = size(q,2);
        x = q(:,1:ninput-1);
        t = q(:,ninput);
        SVMModel = fitcsvm(x,t,'KernelFunction',kernel_function,'Standardize',true, 'IterationLimit',iters_limit,'BoxConstraint',box_constraint);
        ScoreSVMModel = fitPosterior(SVMModel,x,t);
        save ([out,'_trained_net_kfold_',int2str(f)],'ScoreSVMModel');
        fprintf(['\n Trained network for ', int2str(f), '-kfold ', 'SVM Model is saved as: ']);
        fprintf([out,'_trained_net_kfold_',int2str(f),'.mat'],'\n');
        fprintf('\n');
        test = load([out,'_test_data_kfold_',int2str(f),'.txt']);
        ninput = size(test,2);
        xtest = test(:,1:ninput-1);
        ttest = test(:,ninput);
        [label_test,score_test] = predict(ScoreSVMModel,xtest);
        y_svm_test = score_test(:,2); 
        [C_svm]=confmat(y_svm_test,ttest);
        fprintf('\n Confusion Matrix of the Test Model: \r\n');
        disp(C_svm);
        C1 = C_svm(1,1);
        C2 = C_svm(1,2);
        C3 = C_svm(2,1);
        C4 = C_svm(2,2);
        Correct_prediction_SVM = C1+C4;
        Wrong_prediction_SVM = C2+C3;
        Accuracy = (Correct_prediction_SVM/(Correct_prediction_SVM + Wrong_prediction_SVM))*100;
        Sensitivity = C1/(C1+C3);
        Specificity = C4/(C4+C2);
%         disp(Accuracy);
%         disp(Sensitivity);
%         disp(Specificity);
        fprintf('\n Sensitivity of the SVM Test Model = %f\r', Sensitivity);
        fprintf('\n Specificity of the SVM Test Model = %f\r', Specificity);
        fprintf('\n Accuracy of the SVM Test Model = %f\r', Accuracy);
        [faRate, hitRate, AUC] = rocCurve(y_svm_test, ttest);
        fprintf('\n AUC of the Test Model = %f\r\n', AUC);        
        legendinfo{f} = ['AUC = ' num2str(AUC)];
        AUC_COL(f,:) = AUC;
        ACCU(f,:) = Accuracy;
        SENS(f,:) = Sensitivity;
        SPEC(f,:) = Specificity;
        kfold(f,:) = f;
                
    end
    hold off;
    Avg_AUC_test = mean(AUC_COL);
    legend(ah1, (legendinfo),'Location','SouthEast','FontSize',14);
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
    title_svm = title(sprintf(title_svm));
    set(title_svm,'FontSize',16);
    Average_sesitivity = mean(SENS);
    Average_specificity = mean(SPEC);
    Average_accuracy = mean(ACCU);
    fprintf('\n ############ Summary for %d-fold cross validation ############### \n',k);
    summary = [kfold SENS SPEC ACCU AUC_COL];
    TT = array2table(summary,'VariableNames',{'kfold' 'Sensitivity' 'Specificity' 'Accuracy' 'AUC'});
    disp(TT);
    fprintf('\n Average Accuracy of the Test Model = %f\r\n', Average_accuracy);
    Avg_AUC = mean(AUC_COL);
    fprintf('\n Average Sensitivity of the Test Model = %f\r\n', Average_sesitivity);
    fprintf('\n Average Specificity of the Test Model = %f\r\n', Average_specificity);
    fprintf('\n Average AUC of the Test Model = %f\r\n', Avg_AUC);
    kfold_out = [out,'_kfold_ROC'];
    export_fig (kfold_out, '-pdf', '-eps');
    
end