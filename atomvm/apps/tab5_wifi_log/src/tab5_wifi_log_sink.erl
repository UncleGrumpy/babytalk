%% The application's log: the last ?KEEP lines, and every new line pushed to each subscriber
%% (an HTTP request streaming /logs).
%%
%% It is also the listener's subscriber, so the lines the listener's events turn into are
%% exactly listen_demo_responder:handle_info/2's -- same format strings, same reply to a
%% {command, Text} (which is what makes the SAID line happen).
-module(tab5_wifi_log_sink).
-behaviour(gen_server).
-export([start_link/0, log/2, subscribe/1, unsubscribe/1, backlog/0]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).

-define(KEEP, 100). %% lines kept for a client that connects later.

start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

%% One line: to the console (io:format/2, byte for byte what listen_demo_responder prints) and
%% to every stream. Format + args, so a log line can never be mistaken for a format string.
log(Format, Args) ->
    Line = iolist_to_binary(io_lib:format(Format, Args)),
    io:format(Format, Args),
    gen_server:cast(?MODULE, {line, Line}),
    ok.

%% The HTTP handler subscribes for its own pid; it gets {log, Line} messages (backlog first).
subscribe(Pid) ->
    gen_server:call(?MODULE, {subscribe, Pid}).

unsubscribe(Pid) ->
    gen_server:cast(?MODULE, {unsubscribe, Pid}).

%% Sends the accumulated log lines (up to the most recent ?KEEP to the first
%% connected client, then flush - to save as much memory for STT/TTS as possible.
backlog() ->
    gen_server:call(?MODULE, backlog).

init([]) ->
    {ok, {[], []}}.  %% {lines newest first, subscribers}

handle_call({subscribe, Pid}, _From, {Lines, Subs}) ->
    {reply, ok, {Lines, [Pid | Subs]}};
handle_call(backlog, _From, {Lines, Subs}) ->
    {reply, lists:reverse(Lines), {[], Subs}};
handle_call(_Request, _From, S) ->
    {reply, ok, S}.

handle_cast({line, Line}, {Lines, []}) ->
    {noreply, {keep(Line, Lines), []}};
handle_cast({line, Line}, {[], Subs}) ->
    [Pid ! {log, Line} || Pid <- Subs],
    {noreply, {[], Subs}};
handle_cast({line, Line}, {Lines, Subs}) ->
    lists:foreach(fun(Pid) ->
        [Pid ! {log, Line0} || Line0 <- lists:reverse(Lines ++ [Line])] end,
        Subs
    ),
    %[Pid ! {log, [Line || Line <- lists:reverse(Lines ++ [Line])] || Pid <- Subs],
    {noreply, {[], Subs}};
handle_cast({unsubscribe, Pid}, {Lines, Subs}) ->
    {noreply, {Lines, lists:delete(Pid, Subs)}};
handle_cast(_Msg, S) ->
    {noreply, S}.

%% The listener's events: listen_demo_responder:handle_info/2, verbatim
handle_info({babytalk_listener, {heard, Text, Score}}, S) ->
    log("  heard ~p (score ~.4f)\n", [Text, 100.0 + Score]), {noreply, S};
handle_info({babytalk_listener, {wake, _Text, Score}}, S) ->
    log("AWAKE (score ~.4f) -- listening for a command\n", [100 + Score]), {noreply, S};
handle_info({babytalk_listener, {command, Text}}, S) ->
    log("COMMAND ~p\n", [Text]),
    Reply = case Text of
                <<>> -> <<"Sorry, I didn't catch that.">>;
                _ -> [<<"You said: ">>, Text]
            end,
    babytalk_listener:say(babytalk_listener, Reply),
    {noreply, S};
handle_info({babytalk_listener, {said, Text}}, S) ->
    log("SAID ~p\n", [Text]), {noreply, S};
handle_info({babytalk_listener, Other}, S) ->
    log("  ~p\n", [Other]), {noreply, S};
handle_info(_Msg, S) ->
    {noreply, S}.

keep(Line, {Lines, []}) when length(Lines) >= ?KEEP ->
    keep(Line, lists:sublist(Lines, ?KEEP - 1));
keep(_Line, {_Lines, Subs}) when length(Subs) >= 1 ->
    [];
keep(Line, Lines) ->
    [Line | Lines].
