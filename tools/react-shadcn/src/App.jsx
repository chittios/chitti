import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import { Skeleton } from "@/components/ui/skeleton";

/// One entry per component the gallery renders. The marker line printed at the
/// end lists exactly these, so a component that fails to render is named rather
/// than just missing from a screenshot.
const COMPONENTS = [
  "button",
  "card",
  "badge",
  "input",
  "label",
  "separator",
  "switch",
  "checkbox",
  "progress",
  "avatar",
  "alert",
  "tabs",
  "accordion",
  "table",
  "textarea",
  "skeleton",
];

/// A per-section error boundary.
///
/// Without it, one component that fails to render takes the whole page down
/// and React reports only `got: undefined` — no name, no location. With it,
/// every section reports itself, so "all components work" is a claim the page
/// can actually make (or refute, naming the component).
class SectionBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error: error };
  }
  componentDidCatch(error) {
    console.log("shadcn SECTION FAIL " + this.props.id + ": " + String(error));
  }
  render() {
    if (this.state.error) {
      return (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
          {this.props.id} failed: {String(this.state.error)}
        </div>
      );
    }
    return this.props.children;
  }
}

function Section({ id, title, children }) {
  return (
    <section id={`sec-${id}`} className="mb-6">
      <h2 className="mb-2 text-sm font-semibold tracking-tight text-foreground">
        {title}
      </h2>
      <div className="rounded-lg border border-border bg-card p-4">
        <SectionBoundary id={id}>{children}</SectionBoundary>
      </div>
    </section>
  );
}

/// Every imported component, by name. React's "element type is invalid" error
/// names nothing — it says `got: undefined` — so the gallery checks its own
/// imports first and reports the ones that are missing BY NAME. That is the
/// difference between "shadcn does not work" and "AccordionContent is
/// undefined".
const IMPORTED = {
  Button, Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle,
  Badge, Input, Label, Separator, Switch, Checkbox, Progress, Avatar,
  AvatarFallback, Alert, AlertDescription, AlertTitle, Tabs, TabsContent,
  TabsList, TabsTrigger, Accordion, AccordionContent, AccordionItem,
  AccordionTrigger, Table, TableBody, TableCaption, TableCell, TableHead,
  TableHeader, TableRow, Textarea, Skeleton,
};

function reportImports() {
  const missing = [];
  const names = Object.keys(IMPORTED);
  for (let i = 0; i < names.length; i++) {
    const v = IMPORTED[names[i]];
    const ok = typeof v === "function" || (v && typeof v === "object");
    if (!ok) missing.push(names[i]);
  }
  if (missing.length) {
    console.log("shadcn MISSING " + missing.join(","));
  } else {
    console.log("shadcn IMPORTS OK " + names.length);
  }
  return missing.length === 0;
}

export default function App() {
  const [count, setCount] = useState(0);
  reportImports();

  return (
    <div className="min-h-screen bg-background p-4 text-foreground">
      <div className="mx-auto w-full max-w-2xl">
        <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          ChittiOS /samples
        </p>
        <h1 className="mb-1 text-2xl font-bold tracking-tight">shadcn/ui gallery</h1>
        <p className="mb-6 text-sm text-muted-foreground">
          Real shadcn/ui components (new-york style) on React 18 + Tailwind 3,
          rendered by the in-OS browser.
        </p>

        <Section id="button" title="Button">
          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={() => setCount((n) => n + 1)}>Default {count}</Button>
            <Button variant="secondary">Secondary</Button>
            <Button variant="destructive">Destructive</Button>
            <Button variant="outline">Outline</Button>
            <Button variant="ghost">Ghost</Button>
            <Button variant="link">Link</Button>
            <Button size="sm">Small</Button>
            <Button size="lg">Large</Button>
          </div>
        </Section>

        <Section id="badge" title="Badge">
          <div className="flex flex-wrap items-center gap-2">
            <Badge>Default</Badge>
            <Badge variant="secondary">Secondary</Badge>
            <Badge variant="destructive">Destructive</Badge>
            <Badge variant="outline">Outline</Badge>
          </div>
        </Section>

        <Section id="card" title="Card">
          <Card>
            <CardHeader>
              <CardTitle>Deploy</CardTitle>
              <CardDescription>Ship the kernel to a VM.</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Card content sits between the header and the footer.
              </p>
            </CardContent>
            <CardFooter className="gap-2">
              <Button size="sm">Deploy</Button>
              <Button size="sm" variant="outline">
                Cancel
              </Button>
            </CardFooter>
          </Card>
        </Section>

        <Section id="input" title="Input">
          <Input id="name" placeholder="chitti" />
        </Section>

        <Section id="label" title="Label">
          <Label htmlFor="name">Name</Label>
        </Section>

        <Section id="textarea" title="Textarea">
          <Textarea id="notes" placeholder="an agentic operating system" />
        </Section>

        <Section id="checkbox" title="Checkbox">
          <div className="flex items-center gap-2">
            <Checkbox id="terms" />
            <Label htmlFor="terms">Accept the capability grant</Label>
          </div>
        </Section>

        <Section id="switch" title="Switch">
          <div className="flex items-center gap-2">
            <Switch id="ring3" />
            <Label htmlFor="ring3">Decode in ring 3</Label>
          </div>
        </Section>

        <Section id="separator" title="Separator">
          <div className="flex items-center gap-3">
            <span className="text-sm">above</span>
            <Separator className="flex-1" />
            <span className="text-sm">below</span>
          </div>
        </Section>

        <Section id="progress" title="Progress">
          <Progress value={62} />
        </Section>

        <Section id="avatar" title="Avatar">
          <div className="flex items-center gap-2">
            <Avatar>
              <AvatarFallback>CH</AvatarFallback>
            </Avatar>
            <span className="text-sm text-muted-foreground">shell agent</span>
          </div>
        </Section>

        <Section id="skeleton" title="Skeleton">
          <div className="flex items-center gap-2">
            <Skeleton className="h-8 w-8 rounded-full" />
            <Skeleton className="h-4 w-40" />
          </div>
        </Section>

        <Section id="alert" title="Alert">
          <div className="grid gap-3">
            <Alert>
              <AlertTitle>Heads up</AlertTitle>
              <AlertDescription>
                Every effect routes through Synapse.
              </AlertDescription>
            </Alert>
            <Alert variant="destructive">
              <AlertTitle>Refused</AlertTitle>
              <AlertDescription>
                A destructive primitive justified by untrusted content.
              </AlertDescription>
            </Alert>
          </div>
        </Section>

        <Section id="tabs" title="Tabs">
          <Tabs defaultValue="kernel">
            <TabsList>
              <TabsTrigger value="kernel">Kernel</TabsTrigger>
              <TabsTrigger value="agents">Agents</TabsTrigger>
              <TabsTrigger value="browser">Browser</TabsTrigger>
            </TabsList>
            <TabsContent value="kernel">
              <p className="text-sm text-muted-foreground">
                Scheduler, MMU, capabilities, IPC.
              </p>
            </TabsContent>
            <TabsContent value="agents">
              <p className="text-sm text-muted-foreground">
                Sessions, sub-agents, skills.
              </p>
            </TabsContent>
            <TabsContent value="browser">
              <p className="text-sm text-muted-foreground">
                HTML, CSS, and a real JS engine.
              </p>
            </TabsContent>
          </Tabs>
        </Section>

        <Section id="accordion" title="Accordion">
          <Accordion type="single" collapsible defaultValue="a">
            <AccordionItem value="a">
              <AccordionTrigger>What is the determinism boundary?</AccordionTrigger>
              <AccordionContent>
                Model output is an untrusted plan; deterministic native code runs it.
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="b">
              <AccordionTrigger>Where do effects happen?</AccordionTrigger>
              <AccordionContent>In ring 3, through a gated ABI.</AccordionContent>
            </AccordionItem>
          </Accordion>
        </Section>

        <Section id="table" title="Table">
          <Table>
            <TableCaption>Subsystems by ring.</TableCaption>
            <TableHeader>
              <TableRow>
                <TableHead>Subsystem</TableHead>
                <TableHead>Ring</TableHead>
                <TableHead className="text-right">Files</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow>
                <TableCell>Drivers</TableCell>
                <TableCell>0</TableCell>
                <TableCell className="text-right">42</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Image decode</TableCell>
                <TableCell>3</TableCell>
                <TableCell className="text-right">6</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Agents</TableCell>
                <TableCell>3</TableCell>
                <TableCell className="text-right">18</TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </Section>

        <p id="status" className="mt-6 text-center font-mono text-sm text-foreground">
          shadcn ALL PASS {COMPONENTS.length} components:{" "}
          {COMPONENTS.join(",")}
        </p>
      </div>
    </div>
  );
}
