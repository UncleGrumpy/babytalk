%% The HTTP side of the log: an http_server (AtomVM's libs/avm_network) router module.
%%
%%   GET /       the log in a browser, live
%%   GET /logs   the log as Server-Sent Events: curl -N http://<ip>:8080/logs
%%
%% /logs is a chunked response that stays open: the backlog first (one TCP write, each
%% event still its own chunk), then every line as it is logged, one SSE event each, with a
%% comment chunk every ?PING_MS so a dead client is noticed. A failed send ends the stream
%% the same way a closed socket does. http_server:reply/4 sends only the status line and
%% headers (its /3 closes the socket); the chunks are ours, on the same socket, and
%% handle_req stays in this process -- the one that owns it.
-module(tab5_wifi_log_http).
-export([handle_req/3]).

-define(PING_MS, 15000).
-define(PAGE, <<
    "<!doctype html><meta charset=utf-8><title>tab5_wifi_log</title>"
    "<style>body{margin:0;background:#111;color:#ddd;font:13px/1.45 monospace}"
    "pre{padding:12px;white-space:pre-wrap}h3{margin:0;padding:12px;background:#000;color:#7c7}"
    "</style><h3>tab5_wifi_log &mdash; /logs</h3><pre id=\"l\"></pre>"
    "<script>"
    "var l=document.getElementById(\"l\"),es=new EventSource(\"/logs\");"
    "es.onmessage=function(e){l.textContent+=e.data+\"\\n\";window.scrollTo(0,document.body.scrollHeight)};"
    "</script>">>).

handle_req("GET", [], Conn) ->
    http_server:reply(200, ?PAGE, Conn);
handle_req("GET", ["logs"], Conn) ->
    stream(Conn);
handle_req(_Method, _Path, Conn) ->
    http_server:reply(404, <<"not found\n">>, Conn).

stream(Conn) ->
    Socket = proplists:get_value(socket, Conn),
    {ok, _} = http_server:reply(
        200, [],
        <<"Content-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
          "Connection: keep-alive\r\nTransfer-Encoding: chunked\r\n">>,
        Conn
    ),
    ok = tab5_wifi_log_sink:subscribe(self()),
    %% the whole backlog in one TCP write
    case gen_tcp:send(Socket, [event_frame(Line) || Line <- tab5_wifi_log_sink:backlog()]) of
        ok -> follow(Socket, Conn);
        {error, _} -> bye(Conn)
    end.

%% Until the client goes away (or dies): handle_req must return {ok, Conn} either way.
follow(Socket, Conn) ->
    receive
        {log, Line} ->
            case event(Socket, Line) of
                ok -> follow(Socket, Conn);
                {error, _} -> bye(Conn)
            end;
        {tcp_closed, Socket} ->
            bye(Conn);
        {tcp_error, Socket, _} ->
            bye(Conn);
        {tcp, Socket, _Data} ->        %% a client of /logs has nothing to say
            follow(Socket, Conn)
    after ?PING_MS ->
        %% a comment chunk: no event, just proof of life
        case gen_tcp:send(Socket, ["8\r\n: ping\n\n\r\n"]) of
            ok -> follow(Socket, Conn);
            {error, _} -> bye(Conn)
        end
    end.

%% The stream is over (the socket closed or a send failed): stop being a subscriber and
%% let handle_req return.
bye(Conn) ->
    tab5_wifi_log_sink:unsubscribe(self()),
    {ok, Conn}.

%% One log line as an SSE event, in a chunk of its own. The framing is ours, not the log
%% line's: its trailing newline belongs to the console (stripped here), and a newline
%% inside the line is data, which SSE carries as one "data: " line per source line -- the
%% empty line only ever terminates the event. Split on CR too: SSE ends a line at CR as
%% well, so a stray \r must not reach the field body.
event(Socket, Line) ->
    gen_tcp:send(Socket, event_frame(Line)).

event_frame(Line) ->
    Bin = iolist_to_binary(Line),
    Body = strip_console_lf(Bin),
    Data = [<<"data: ", P/binary, "\n">>
            || P <- binary:split(Body, [<<"\r\n">>, <<"\n">>, <<"\r">>], [global])],
    Chunk = iolist_to_binary([Data, "\n"]),
    [integer_to_list(byte_size(Chunk), 16), "\r\n", Chunk, "\r\n"].

%% log/2's format strings keep their ~n for the console; the SSE terminator is added here.
strip_console_lf(<<>>) -> <<>>;
strip_console_lf(Bin) ->
    case binary:last(Bin) of
        $\n -> binary:part(Bin, 0, byte_size(Bin) - 1);
        _ -> Bin
    end.
