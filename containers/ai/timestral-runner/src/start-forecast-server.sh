#!/bin/sh

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

set -eu

exec python -m uvicorn forecast_server:app --host 0.0.0.0 --port "${TIMESTRAL_PORT:-1345}" --log-level "${TIMESTRAL_LOG_LEVEL:-warning}"
