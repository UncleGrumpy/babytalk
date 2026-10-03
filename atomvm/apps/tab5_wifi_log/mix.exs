defmodule Tab5WifiLog.MixProject do
  use Mix.Project

  def project do
    [
      app: :tab5_wifi_log,
      version: "0.1.0",
      elixir: "~> 1.17",
      erlc_paths: ["src", "../../lib/babytalk/src"],
      deps: [
        {:exatomvm, git: "https://github.com/atomvm/exatomvm.git", runtime: false},
        {:atomvm, "~> 0.7.0-beta.0", runtime: false},
        {:pythonx, "~> 0.4.0"}
      ],
      atomvm: [start: :tab5_wifi_log, flash_offset: 0xA90000]
    ]
  end

  def application, do: []
end
