%% The HTTP side of the log: an http_server (AtomVM's libs/avm_network) router module.
%%
%%   GET /       the log in a browser, live
%%   GET /logs   the log as Server-Sent Events: curl -N http://<ip>:8080/logs
%%
%% /logs is a chunked response that stays open: the backlog first, then every line as it is
%% logged, one SSE event each, with a comment chunk every ?PING_MS so a dead client is noticed.
%% http_server:reply/4 sends only the status line and headers (its /3 closes the socket); the
%% chunks are ours, on the same socket, and handle_req stays in this process -- the one that
%% owns it.
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
    [event(Socket, Line) || Line <- tab5_wifi_log_sink:backlog()],
    follow(Socket, Conn).

%% Until the client goes away (or dies): handle_req must return {ok, Conn} either way.
follow(Socket, Conn) ->
    receive
        {log, Line} ->
            event(Socket, Line),
            follow(Socket, Conn);
        {tcp_closed, Socket} ->
            tab5_wifi_log_sink:unsubscribe(self()),
            {ok, Conn};
        {tcp, Socket, _Data} ->        %% a client of /logs has nothing to say
            follow(Socket, Conn)
    after ?PING_MS ->
        %% a comment chunk: no event, just proof of life
        gen_tcp:send(Socket, ["8\r\n: ping\n\n\r\n"]),
        follow(Socket, Conn)
    end.

%% One log line as an SSE event, in a chunk of its own: "data: <line>\n\n"
event(Socket, Line) ->
    Bin = <<"data: ", (iolist_to_binary(Line))/binary, "\n">>,
    gen_tcp:send(Socket, [integer_to_list(byte_size(Bin), 16), "\r\n", Bin, "\r\n"]).
