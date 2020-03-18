%% Compare RFLO learning, RTRL, and BPTT for producing a simple periodic output.

[n_in, n_rec, n_out] = deal(1, 50, 1);  % number of inputs, recurrent units, and outputs
duration = 100;  % number of timesteps in one period

% Input to RNN
x_tonic = 0.0*ones(duration, n_in);

% Target RNN output:
y_target = (sin(2*pi*(1:duration)/duration) + ...
                    0.5*sin(2*2*pi*(1:duration)/duration) + ...
                    0.25*sin(4*2*pi*(1:duration)/duration))'*ones(1,n_out);

n_tr = 10000;                   % number of trials to train on
h_init = 0.1*ones(n_rec,1);     % initial state of the RNN

%% Train networks.

fprintf('\nTraining with BPTT...\n');
net1 = rnn(n_in, n_rec, n_out, h_init);
learn_rates = [0.0, 0.03, 0.03];
[y_pre1, y_post1, loss_list1, ~] = ...
    net1.run_session(n_tr, x_tonic, y_target, learn_rates, 'bptt', false);

fprintf('\nTraining with RFLO...\n');
net2 = rnn(n_in, n_rec, n_out, h_init);
learn_rates = [0.0, 0.03, 0.03];
[y_pre2, y_post2, loss_list2, fbalign] = ...
    net2.run_session(n_tr, x_tonic, y_target, learn_rates,'rflo', false);

% Skip training with RTRL because it's very slow.
%print('\nTraining with RTRL...')
%net3 = RNN(n_in, n_rec, n_out, h_init)
%learn_rates = [0.0, 0.03, 0.03]
%[~, loss_list3, ~] = ...
%    net3.run_session(n_tr, x_tonic, y_target, learn_rates,'rtrl', false);

%% Plot the results.

% Plot learning curve
figure;

subplot(121);
loglog(loss_list1, 'b');
title('BPTT','Fontsize',16);
ylabel('Error','Fontsize',16);
xlabel('Trial','Fontsize',16);

subplot(122);
loglog(loss_list2, 'b');
title('RFLO','Fontsize',16);
ylabel('Error','Fontsize',16);
xlabel('Trial','Fontsize',16);

%subplot(121);
%loglog(loss_list3, 'b');
%title('RTRL');
%xlabel('Trial');


% Test the networks with learning turned off:
y1 = net1.run_trial(x_tonic, y_target, [0.1 0.1 0.1], false, false);
y2 = net2.run_trial(x_tonic, y_target, [0.1 0.1 0.1], false, false);
% y3 = net3.run_trial(x_tonic, y_target, false);


% Plot the RNN output along with the target output:
figure;

subplot(121); hold on;
plot(y_pre1); 
plot(y_post1);
plot(y_target, '--k');
legend('pre-training','post-training','target','Fontsize',16);
title('BPTT','Fontsize',16);
ylabel('y(t)','Fontsize',16);
xlabel('Time, t','Fontsize',16);

subplot(122); hold on;
plot(y_pre2); 
plot(y_post2);
plot(y_target, '--k');
title('RFLO','Fontsize',16);
xlabel('Time, t','Fontsize',16);

%subplot(121)
%plot(y3);
%plot(y_target, '--k');
%title('RTRL');
%xlabel('Time');

% Plot feedback alignment
figure;
loglog(fbalign);
title('RFLO','Fontsize',16);
ylabel('Cosine Similarity (W^{out},B)','Fontsize',16);
xlabel('Time','Fontsize',16);