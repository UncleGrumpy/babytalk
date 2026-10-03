%% The listener that produces the log, as listen_demo's supervisor starts it: the same wake
%% phrase and threshold (listen_demo_sup), the board preset, and our log sink as the subscriber.
%%
%% one_for_one, not rest_for_one: AtomVM's supervisor implements only one_for_one and
%% one_for_all (supervisor:handle_restart_strategy/2 has no rest_for_one clause).
-module(tab5_wifi_log_sup).
-behaviour(supervisor).
-export([start_link/0, init/1]).

%% From listen_demo_sup: tools/wake_phrases/wake_up_tomato_face.json, 98% detection at -21
-define(SPELLINGS, [<<"wake up tomato face">>, <<"wake up to mato face">>, <<"wake up to meato face">>,
                    <<"wake up to meatto face">>, <<"wake up to meadow face">>, <<"wake up t martr face">>,
                    <<"wake up to marto face from">>, <<"wake up to matle face">>, <<"wake up to midto face">>,
                    <<"wake up tomato faceo">>, <<"wake up tomato face a">>, <<"wake up tomato face f">>]).

start_link() ->
    supervisor:start_link({local, ?MODULE}, ?MODULE, []).

init([]) ->
    Listener = #{
        notify => tab5_wifi_log_sink,
        audio => m5stack_tab5,
        spellings => ?SPELLINGS,
        threshold => -21.0
    },
    {ok, {#{strategy => one_for_one, intensity => 5, period => 60},
          [#{id => listener, start => {babytalk_listener, start_link, [babytalk_listener, Listener]}}]}}.
