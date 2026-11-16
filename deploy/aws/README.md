# The console demo on AWS

`demo-site.yaml` is a CloudFormation stack that hosts the static console demo
([docs/07-deployment.md](../../docs/07-deployment.md#the-console-demo)) at https://dev.meritorder.dev: a private S3
bucket, CloudFront with Origin Access Control and security headers, an ACM certificate, Route 53 alias records, and an
IAM role GitHub Actions deploys with. No servers. Plain CloudFormation, no other tooling.

It lives in Merit Order's own AWS account. Use that account's profile and nothing else.

## Create or change the stack

The stack goes in **us-east-1**: CloudFront only takes certificates from there.

```bash
aws sts get-caller-identity --profile meritorder-admin        # the Merit Order account
aws route53 list-hosted-zones-by-name --dns-name meritorder.dev --profile meritorder-admin   # the zone id
aws iam list-open-id-connect-providers --profile meritorder-admin   # a GitHub provider already there? then pass false below

aws cloudformation deploy --profile meritorder-admin --region us-east-1 \
  --stack-name meritorder-demo-site --template-file deploy/aws/demo-site.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides HostedZoneId=<zone id> CreateGitHubOidcProvider=true \
  --tags project=meritorder purpose=console-demo
```

The first create takes about 5 to 15 minutes: CloudFormation writes the certificate's validation record into the zone,
waits for ACM to issue it, then waits for the distribution. A change to the headers or behaviors takes a few minutes
to reach every edge.

Parameters:

| Parameter | Default | What it is |
|---|---|---|
| `DomainName` | `dev.meritorder.dev` | The name the demo is served at. It must be in the zone. |
| `HostedZoneId` | (required) | The Route 53 zone for `meritorder.dev`. |
| `GitHubSubjects` | `repo:therick@319326/meritorder@1451524709:ref:refs/heads/main,repo:therick/meritorder:ref:refs/heads/main,repo:Merit-Order/meritorder:ref:refs/heads/main` | Who the deploy role trusts. Repos with GitHub's immutable subjects (`gh api repos/<owner>/<repo>/actions/oidc/customization/sub`) send `repo:<owner>@<id>/<repo>@<id>:...`; list that form. |
| `CreateGitHubOidcProvider` | `true` | An account has one OIDC provider for `token.actions.githubusercontent.com`. `false` uses the existing one. |

Outputs: `BucketName`, `DistributionId`, `DistributionDomainName`, `DeployRoleArn`, `Url`.

```bash
aws cloudformation describe-stacks --profile meritorder-admin --region us-east-1 \
  --stack-name meritorder-demo-site --query 'Stacks[0].Outputs' --output table
```

## The deploy role

- **Trust:** `sts:AssumeRoleWithWebIdentity` from GitHub's OIDC provider, when the token's audience is
  `sts.amazonaws.com` and its subject is one of `GitHubSubjects`: a workflow on `main` of `therick/meritorder` or
  `Merit-Order/meritorder`. Branches, tags, pull requests and jobs that name an environment get other subjects and are
  refused.
- **Permissions:** `s3:ListBucket` on the bucket, `s3:PutObject` and `s3:DeleteObject` on its objects, and
  `cloudfront:CreateInvalidation` on the distribution. Nothing else.

## Repository variables

The pipeline (`.github/workflows/demo.yml`) reads three repository variables. They aren't secrets.

```bash
gh variable set DEMO_DEPLOY_ROLE_ARN --repo Merit-Order/meritorder --body <DeployRoleArn>
gh variable set DEMO_BUCKET          --repo Merit-Order/meritorder --body <BucketName>
gh variable set DEMO_DISTRIBUTION_ID --repo Merit-Order/meritorder --body <DistributionId>
```

The role already trusts pushes to `main` of `Merit-Order/meritorder` by name. Once the repository exists, add its
immutable subject (`repo:Merit-Order@<owner id>/meritorder@<repo id>:ref:refs/heads/main`) to `GitHubSubjects` if it
sends that form.

## Deploying by hand

What the pipeline does, from a checkout:

```bash
uv run python scripts/build_demo_data.py
cd ui && npm ci && npm run build:demo
B=<BucketName>; D=<DistributionId>; P="--profile meritorder-admin"
aws s3 sync dist-demo/assets s3://$B/assets --cache-control "public, max-age=31536000, immutable" $P
aws s3 sync dist-demo s3://$B --exclude "assets/*" --exclude ".nojekyll" --cache-control "no-cache" --delete $P
aws s3 sync dist-demo/assets s3://$B/assets --cache-control "public, max-age=31536000, immutable" --delete $P
aws cloudfront create-invalidation --distribution-id $D --paths "/*" $P
DEMO_URL=https://dev.meritorder.dev/ npm run e2e:demo     # the smoke test against the live site
```

## Costs

About $0.50 a month at a demo's traffic: the Route 53 hosted zone. CloudFront's free tier (1 TB and 10 million
requests a month) covers the traffic, S3 storage is a few MB, the certificate is free, and one `/*` invalidation per
deploy is well inside the 1,000 free paths a month. See
[docs/07-deployment.md](../../docs/07-deployment.md#hosting-it-devmeritorderdev-on-aws) for the numbers.

## Removing it

Empty the bucket (`aws s3 rm s3://<BucketName> --recursive`), then
`aws cloudformation delete-stack --stack-name meritorder-demo-site --region us-east-1`. Deleting the stack also removes
the GitHub OIDC provider if the stack created it.
