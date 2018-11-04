fprintf('\nExtracting features..\n');

%3051 is the comprehensive numbers of spam mails + non-spam mails (now I use 10 for testing)

X = zeros(10, 1899);
y = zeros(10, 1);




% Extract 5 Features non-spam


%file 1
file_contents = readFile('spamAssassinPublicCorpus/nonSpamTest/1');
word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices);
X(1,:) = features;

%file 2
file_contents = readFile('spamAssassinPublicCorpus/nonSpamTest/2');
word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices);
X(2,:) = features;

%file 3
file_contents = readFile('spamAssassinPublicCorpus/nonSpamTest/3');
word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices);
X(3,:) = features;

%file 4
file_contents = readFile('spamAssassinPublicCorpus/nonSpamTest/4');
word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices);
X(4,:) = features;

%file 5
file_contents = readFile('spamAssassinPublicCorpus/nonSpamTest/5');
word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices);
X(5,:) = features;



% Extract 5 Features spam


%file 1
file_contents = readFile('spamAssassinPublicCorpus/spamTest/1');
word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices);
X(6,:) = features;
y(6) = 1;

%file 2
file_contents = readFile('spamAssassinPublicCorpus/spamTest/2');
word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices);
X(7,:) = features;
y(7) = 1;

%file 3
file_contents = readFile('spamAssassinPublicCorpus/spamTest/3');
word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices);
X(8,:) = features;
y(8) = 1;

%file 4
file_contents = readFile('spamAssassinPublicCorpus/spamTest/4');
word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices);
X(9,:) = features;
y(9) = 1;

%file 5
file_contents = readFile('spamAssassinPublicCorpus/spamTest/5');
word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices);
X(10,:) = features;
y(10) = 1;




%Training
C = 0.1;
model = svmTrain(X, y, C, @linearKernel);

p = svmPredict(model, X);

fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);





%Testing on test set

Xtest = zeros(3, 1899);
ytest = zeros(3, 1);



%file 11
file_contents = readFile('spamAssassinPublicCorpus/nonSpamTest/11');
word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices);
Xtest(1,:) = features;

%file 12
file_contents = readFile('spamAssassinPublicCorpus/nonSpamTest/12');
word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices);
Xtest(2,:) = features;

%file 13
file_contents = readFile('spamAssassinPublicCorpus/nonSpamTest/13');
word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices);
Xtest(3,:) = features;






%file 14
file_contents = readFile('spamAssassinPublicCorpus/spamTest/14');
word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices);
Xtest(4,:) = features;
ytest(4) = 1;


%file 15
file_contents = readFile('spamAssassinPublicCorpus/spamTest/15');
word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices);
Xtest(5,:) = features;
ytest(5) = 1;



%file 16
file_contents = readFile('spamAssassinPublicCorpus/spamTest/16');
word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices);
Xtest(6,:) = features;
ytest(6) = 1;


p = svmPredict(model, Xtest);

fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);
