classdef rnn < handle
%{
    A recurrent neural network.

    Parameters:
    ----------
    n_in, n_rec, n_out : number of input, recurrent, and hidden units.
    h0 : The initial state vector of the RNN.
    tau_m : The network time constant, in units of timesteps.
%}
    %%
    properties
        n_in
        n_rec
        n_out
        h0
        tau_m
        w_in
        w_rec
        w_out
        b
    end
    %%
    methods
        %% class constructor
        function this = rnn(n_in, n_rec, n_out, h0, tau_m)
            if nargin<5, tau_m = 10; end
            this.n_in = n_in;
            this.n_rec = n_rec;
            this.n_out = n_out;
            this.h0 = h0;
            this.tau_m = tau_m;
            this.w_in = 0.1*(rand(n_rec,n_in) - 1);
            this.w_rec = 1.5*randn(n_rec, n_rec)/sqrt(n_rec);
            this.w_out = 0.1*(2*rand(n_out, n_rec) - 1)/sqrt(n_rec);
            this.b = randn(n_rec, n_out)/sqrt(n_out);
        end
    end
end