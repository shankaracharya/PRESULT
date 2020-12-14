function RT_kfold(varargin)
fprintf('\n Example command line: "./presult kfold RT --input_data <input_data.txt> --randm <yes/no> --kfold <int>" \n')
p = inputParser;
addParameter(p,'input_data','pid_raw_data.txt',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'kfold','10');
addParameter(p,'var','variable.txt',@ischar);
addParameter(p,'out','RT',@ischar);
addParameter(p,'min_parent','50');
addParameter(p,'min_leaf','12');
addParameter(p,'randm','shuffle',@ischar);
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

new_array(strcmp('yes',new_array))={'shuffle'};
new_array(strcmp('no',new_array))={'default'};

parse(p,new_array{:});
data_file = (p.Results.input_data);
kvalue = (p.Results.kfold);
min_parent = (p.Results.min_parent);
min_leaf = (p.Results.min_leaf);
csv = (p.Results.csv);
var = (p.Results.var);
miss = (p.Results.missing_value);
out = (p.Results.out);
ran = (p.Results.randm);
title_rt = (p.Results.title);

if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', data_file, '--output', 'normalized_data_kfold_RT.txt','--var',var);
else
   perl('newest_normalization.pl', '--input', data_file, '--output', 'normalized_data_kfold_RT.txt', '--csv','--var',var);
end
data_file1 = load('normalized_data_kfold_RT.txt');
k = str2double(kvalue);
min_parent = str2double(min_parent);
min_leaf = str2double(min_leaf);


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
%k = 10;
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
        rtree = fitrtree(x,t,'MinParent',min_parent,'MinLeaf', min_leaf,'MergeLeaves','on','NumVariablesToSample','all','Prune','off','Surrogate','all');
        test = load([out,'_test_data_kfold_',int2str(f),'.txt']);
        save ([out,'_trained_net_kfold_',int2str(f)],'rtree');
        fprintf(['\n Trained network for ', int2str(f), '-kfold ', 'RF Model is saved as: ']);
        fprintf([out,'_trained_net_kfold_',int2str(f),'.mat'],'\n');
        fprintf('\n');
        ninput = size(test,2);
        xtest = test(:,1:ninput-1);
        ttest = test(:,ninput);
       
        y_test = predict(rtree,xtest);
        
        [C]=confmat(y_test,ttest);
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
        [faRate, hitRate, AUC] = rocCurve(y_test, ttest);
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
    xlabel_kf = xlabel('1-Specificity');
    set(xlabel_kf,'FontSize',16);
    ylabel_kf = ylabel('Sensitivity');
    set(ylabel_kf,'FontSize',16);
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    box on;
    dim =[0.63 0.4 0.1 0.2];
    str = ['Avg AUC = ',num2str(Avg_AUC_test)];
    annotation('textbox',dim,'String',str,'FitBoxToText','on','FontSize',14,'Color','k');
    Average_sesitivity = mean(SENS);
    Average_specificity = mean(SPEC);
    Average_accuracy = mean(ACCU);
    fprintf('\n ############ Summary for %d-fold cross validation ############### \n',k);
    summary = [kfold SENS SPEC ACCU AUC_COL];
    TT = array2table(summary,'VariableNames',{'kfold' 'Sensitivity' 'Specificity' 'Accuracy' 'AUC'});
    disp(TT);
    fprintf('\n Average Accuracy of the Test Model = %f\r\n', Average_accuracy);
    Avg_AUC = mean(AUC_COL);
    title_rt = title(ah1, sprintf(title_rt));
    set(title_rt,'FontSize',16);
    fprintf('\n Average Sensitivity of the Test Model = %f\r\n', Average_sesitivity);
    fprintf('\n Average Specificity of the Test Model = %f\r\n', Average_specificity);
    fprintf('\n Average AUC of the Test Model = %f\r\n', Avg_AUC);
    kfold_out = [out,'_kfold_ROC'];
    export_fig (kfold_out, '-pdf', '-eps');
    
end