{{- define "meritorder.fullname" -}}
{{- if contains .Chart.Name .Release.Name -}}{{ .Release.Name | trunc 63 | trimSuffix "-" }}{{- else -}}{{ printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" }}{{- end -}}
{{- end -}}

{{- define "meritorder.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{- define "meritorder.selector" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "meritorder.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}
{{- end -}}

{{- define "meritorder.secretName" -}}
{{- if .Values.existingSecret -}}{{ .Values.existingSecret }}{{- else -}}{{ include "meritorder.fullname" . }}-env{{- end -}}
{{- end -}}

{{/* "true" when some Secret feeds the environment; fails if more than one source is configured. */}}
{{- define "meritorder.hasSecret" -}}
{{- $n := 0 -}}
{{- if .Values.existingSecret }}{{ $n = add1 $n }}{{ end -}}
{{- if .Values.secretEnv }}{{ $n = add1 $n }}{{ end -}}
{{- if .Values.externalSecret.enabled }}{{ $n = add1 $n }}{{ end -}}
{{- if gt $n 1 }}{{ fail "set only one of externalSecret.enabled, existingSecret, and secretEnv" }}{{ end -}}
{{- if gt $n 0 }}true{{ end -}}
{{- end -}}

{{/* Deployment metadata annotations: let Reloader roll the pods when the Secret changes. */}}
{{- define "meritorder.deploymentAnnotations" -}}
{{- if and .Values.secrets.reloadOnChange (include "meritorder.hasSecret" .) }}
annotations:
  secret.reloader.stakater.com/reload: {{ include "meritorder.secretName" . | quote }}
{{- end }}
{{- end -}}

{{/* Pod template metadata: labels and the checksums that roll pods on chart-managed changes. */}}
{{- define "meritorder.podMeta" -}}
{{- with .Values.podLabels }}
{{ toYaml . }}
{{- end }}
{{- end -}}

{{/* Shared pod spec pieces: env, config mounts, security.
     Called with (dict "root" $ "hook" true|false), plus "serviceAccount" for a pod with its own (the
     controller, when it runs GPUs on this cluster). Hook pods (migrations) run before the release's
     ServiceAccount exists, so they use the namespace default. */}}
{{- define "meritorder.podCommon" -}}
{{- $hook := .hook -}}
{{- $own := .serviceAccount -}}
{{- $sky := .skypilot -}}
{{- with .root -}}
{{- if $own }}
serviceAccountName: {{ $own }}
{{- else if and .Values.serviceAccount.create (not $hook) }}
serviceAccountName: {{ include "meritorder.fullname" . }}
{{- else }}
automountServiceAccountToken: false
{{- end }}
securityContext: {{- toYaml .Values.podSecurityContext | nindent 2 }}
{{- with .Values.nodeSelector }}
nodeSelector: {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .Values.tolerations }}
tolerations: {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .Values.affinity }}
affinity: {{- toYaml . | nindent 2 }}
{{- end }}
volumes:
  - name: tmp
    emptyDir: {}
  {{- if and .Values.secrets.mountAsFiles (include "meritorder.hasSecret" .) }}
  - name: secret-files
    secret:
      secretName: {{ include "meritorder.secretName" . }}
      defaultMode: 0440
  {{- end }}
  {{- if .Values.configFiles }}
  - name: config
    configMap:
      name: {{ include "meritorder.fullname" . }}-config
  {{- end }}
  {{- if $sky }}
  {{- /* SkyPilot's home (~/.sky and the SSH configs it writes), and its cloud credentials. */}}
  - name: skypilot-home
    {{- if $sky.home.existingClaim }}
    persistentVolumeClaim:
      claimName: {{ $sky.home.existingClaim }}
    {{- else if $sky.home.persistence }}
    persistentVolumeClaim:
      claimName: {{ include "meritorder.fullname" . }}-skypilot-home
    {{- else }}
    emptyDir: {}
    {{- end }}
  {{- range $i, $c := $sky.credentials }}
  - name: skypilot-credentials-{{ $i }}
    secret:
      secretName: {{ required "each controller.skypilot.credentials entry needs a secret" $c.secret }}
      defaultMode: 0400
  {{- end }}
  {{- end }}
{{- end -}}
{{- end -}}

{{/* Shared container pieces. Called with the root context, or (dict "root" $ "env" (dict NAME value))
     for environment one component alone gets. */}}
{{- define "meritorder.containerCommon" -}}
{{- $extra := dict -}}
{{- $root := . -}}
{{- $sky := "" -}}
{{- if hasKey . "root" }}{{ $root = .root }}{{ $extra = .env | default dict }}{{ $sky = .skypilot | default "" }}{{ end -}}
{{- with $root -}}
imagePullPolicy: {{ .Values.image.pullPolicy }}
securityContext: {{- toYaml .Values.securityContext | nindent 2 }}
envFrom:
  {{- if include "meritorder.hasSecret" . }}
  - secretRef:
      name: {{ include "meritorder.secretName" . }}
  {{- end }}
env:
  - name: MERITORDER_STATE_DIR
    value: /tmp/meritorder
  {{- if not (hasKey .Values.env "MERITORDER_GATEWAY_URL") }}
  # The worker runs console-started evals through the gateway, by its Service.
  - name: MERITORDER_GATEWAY_URL
    value: {{ printf "http://%s:%v" (include "meritorder.fullname" .) .Values.service.port | quote }}
  {{- end }}
  {{- if .Values.compressor.enabled }}
  {{- /* Model-based compression's service (compressor.yaml). Every component gets these: they all validate the
       config, and a team with compression.model on needs compressor.url (the shipped config reads it from here). */}}
  {{- if not (hasKey .Values.env "MERITORDER_COMPRESSOR_URL") }}
  - name: MERITORDER_COMPRESSOR_URL
    value: {{ printf "http://%s-compressor:%v" (include "meritorder.fullname" .) .Values.compressor.port | quote }}
  {{- end }}
  {{- if not (hasKey .Values.env "MERITORDER_COMPRESSOR_MODEL") }}
  - name: MERITORDER_COMPRESSOR_MODEL
    value: {{ .Values.compressor.model | quote }}
  {{- end }}
  {{- end }}
  {{- with .Values.reporting.clickhouse }}
  {{- if .url }}
  # The ledger's copy in ClickHouse (config: reporting.clickhouse); the password comes from the env Secret.
  - name: MERITORDER_CLICKHOUSE_URL
    value: {{ .url | quote }}
  - name: MERITORDER_CLICKHOUSE_DATABASE
    value: {{ .database | default "meritorder" | quote }}
  - name: MERITORDER_CLICKHOUSE_USER
    value: {{ .user | default "default" | quote }}
  {{- end }}
  {{- end }}
  {{- range $k, $v := .Values.env }}
  - name: {{ $k }}
    value: {{ $v | quote }}
  {{- end }}
  {{- range $k, $v := $extra }}
  {{- if not (hasKey $root.Values.env $k) }}
  - name: {{ $k }}
    value: {{ $v | quote }}
  {{- end }}
  {{- end }}
volumeMounts:
  - name: tmp
    mountPath: /tmp
  {{- if and .Values.secrets.mountAsFiles (include "meritorder.hasSecret" .) }}
  - name: secret-files
    mountPath: {{ .Values.secrets.mountPath }}
    readOnly: true
  {{- end }}
  {{- range $name, $_ := .Values.configFiles }}
  - name: config
    mountPath: /app/config/{{ $name }}
    subPath: {{ $name }}
  {{- end }}
  {{- if $sky }}
  - name: skypilot-home
    mountPath: /home/meritorder
  {{- range $i, $c := $sky.credentials }}
  - name: skypilot-credentials-{{ $i }}
    mountPath: /home/meritorder/{{ required "each controller.skypilot.credentials entry needs a path" $c.path | trimPrefix "/" }}
    readOnly: true
  {{- end }}
  {{- end }}
{{- end -}}
{{- end -}}

{{/* The controller's own ServiceAccount, when it runs GPUs on this cluster. */}}
{{- define "meritorder.controllerServiceAccount" -}}
{{ include "meritorder.fullname" . }}-controller
{{- end -}}

{{/* Namespaces GPU deployments run in, as a JSON list: controller.kubernetes.namespaces, else the release's. */}}
{{- define "meritorder.gpuNamespaces" -}}
{{- toJson (.Values.controller.kubernetes.namespaces | default (list .Release.Namespace)) -}}
{{- end -}}
